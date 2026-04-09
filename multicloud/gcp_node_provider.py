import copy
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

from googleapiclient import discovery
import google.auth

from ray.autoscaler._private.gcp.node_provider import (
    GCPNodeProvider as RayGcpNodeProvider,
)
from ray.autoscaler._private.slurm.node_provider import SlurmClusterState

logger = logging.getLogger(__name__)


class GcpNodeProvider:
    def __init__(
        self,
        provider_config: Dict[str, Any],
        cluster_name: str,
        slurm_cluster_state: SlurmClusterState,
    ) -> None:
        # Note that provider_config["auth"] is filled by _bootstrap_scoped in MulticloudNodeProvider.
        self.provider_config = provider_config

        self.cluster_name = cluster_name
        self.slurm_cluster_state = slurm_cluster_state

        self._delegate = RayGcpNodeProvider(provider_config, cluster_name)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)

        if "gcp" in config["provider"]:
            # Fix config for bootstrapping by moving some fields to top level
            config["auth"]["ssh_user"] = config["provider"]["gcp"]["auth"]["ssh_user"]
            provider_keys = config["provider"]["gcp"].keys()
            for k in provider_keys:
                config["provider"][k] = config["provider"]["gcp"][k]

            # Run bootstrap config
            config = RayGcpNodeProvider.bootstrap_config(config)

            # Undo fixes
            config["auth"].pop("ssh_user")
            for k in provider_keys:
                config["provider"].pop(k)

        return config

    @staticmethod
    def fillout_available_node_types_resources(
        cluster_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = RayGcpNodeProvider.fillout_available_node_types_resources(config)
        return config

    def prepare_for_head_node(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._delegate.prepare_for_head_node(cluster_config)

    @property
    def max_terminate_nodes(self) -> Optional[int]:
        return self._delegate.max_terminate_nodes

    def is_readonly(self) -> bool:
        return self._delegate.is_readonly()

    def create_node(
        self, node_config: Dict[str, Any], tags: Dict[str, str], count: int
    ) -> Optional[Dict[str, Any]]:
        res = self._delegate.create_node(node_config, tags, count)

        meta_info = self.slurm_cluster_state.get_meta_info()

        # TODO below steps block autoscaler
        for node_id in res.keys():
            print(f"Waiting for node {node_id} init")
            while not self.is_running(node_id):
                time.sleep(10)
            print(f"Node {node_id} is running")
            node_ip = self.external_ip(node_id)
            assert node_ip is not None

            while True:
                ports = [6379, 10001, 7000, 7001, 7002]
                ports.extend(range(10002, 10100))  # TODO hacky, should be configurable

                tunnel_cmd = [
                    "ssh",
                    "-i",
                    self.provider_config["auth"]["ssh_private_key"],
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "ConnectTimeout=10",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ServerAliveInterval=5",
                    "-o",
                    "ServerAliveCountMax=3",
                    "-f",
                    "-N",
                ]

                for port in ports:
                    tunnel_cmd.append("-R")
                    tunnel_cmd.append(f"{port}:localhost:{port}")

                tunnel_cmd.append(
                    f"{self.provider_config['auth']['ssh_user']}@{node_ip}"
                )

                try:
                    print(f"Run SSH tunneling command for {node_id}\n")
                    subprocess.run(tunnel_cmd, check=True)
                    print(f"SSH tunnel setup for {node_id}\n")
                    break
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        f"SSH tunneling command failed for {node_id}: " + str(e)
                    )
                    time.sleep(10)

            ray_start_command = "ray start"
            ray_start_command += ' --address="localhost:6379"'
            ray_start_command += ' --node-ip-address="' + node_ip + '"'
            ray_start_command += (
                ' --redis-password="' + meta_info["redis_password"] + '"'
            )

            logger.info(f"Run init command ({ray_start_command})\n")
            self.get_command_runner(
                "GcpNodeProvider create:",
                node_id,
                {},
                self.cluster_name,
                subprocess,
                False,
            ).run(ray_start_command)

        return res

    def create_node_with_resources_and_labels(
        self,
        node_config: Dict[str, Any],
        tags: Dict[str, str],
        count: int,
        resources: Dict[str, float],
        labels: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        return self._delegate.create_node_with_resources_and_labels(
            node_config, tags, count, resources, labels
        )

    def terminate_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self._delegate.terminate_node(node_id)

    def terminate_nodes(self, node_ids: List[str]) -> Optional[Dict[str, Any]]:
        return self._delegate.terminate_nodes(node_ids)

    def non_terminated_nodes(self, tag_filters: Dict[str, str]) -> List[str]:
        return self._delegate.non_terminated_nodes(tag_filters)

    def is_running(self, node_id: str) -> bool:
        return self._delegate.is_running(node_id)

    def is_terminated(self, node_id: str) -> bool:
        return self._delegate.is_terminated(node_id)

    def set_node_tags(self, node_id: str, tags: Dict[str, str]) -> None:
        self._delegate.set_node_tags(node_id, tags)

    def node_tags(self, node_id: str) -> Dict[str, str]:
        return self._delegate.node_tags(node_id)

    def external_ip(self, node_id: str) -> Optional[str]:
        return self._delegate.external_ip(node_id)

    def internal_ip(self, node_id: str) -> Optional[str]:
        # Force use of external IP since we're tunneling
        return self._delegate.external_ip(node_id)

    def get_node_id(self, ip_address: str, use_internal_ip: bool = True) -> str:
        return self._delegate.get_node_id(ip_address, use_internal_ip)

    def safe_to_scale(self) -> bool:
        return self._delegate.safe_to_scale()

    def post_process(self) -> None:
        return self._delegate.post_process()

    def get_command_runner(
        self,
        log_prefix: str,
        node_id: str,
        auth_config: Dict[str, Any],
        cluster_name: str,
        process_runner,
        use_internal_ip: bool,
        docker_config: Optional[Dict[str, Any]] = None,
    ):
        use_internal_ip = False  # TODO hacky
        auth_config = {**self.provider_config["auth"], **auth_config}
        return self._delegate.get_command_runner(
            log_prefix,
            node_id,
            auth_config,
            cluster_name,
            process_runner,
            use_internal_ip,
            docker_config,
        )

    def get_spot_rate(self, provider_machine_type: str) -> float:
        with open('~/gcp_price.txt') as f:
            return int(f.read())
        # machine_type = provider_machine_type
        # region = self.provider_config["region"]
        # zone = self.provider_config["availability_zone"]

        # credentials, project = google.auth.default()

        # # Get machine type specs (vCPUs and memory)
        # compute = discovery.build("compute", "v1", credentials=credentials)
        # mt = (
        #     compute.machineTypes()
        #     .get(project=project, zone=zone, machineType=machine_type)
        #     .execute()
        # )
        # vcpus = mt["guestCpus"]
        # memory_gb = mt["memoryMb"] / 1024

        # family = machine_type.split("-")[0].upper()  # "e2-small" -> "E2"

        # billing = discovery.build("cloudbilling", "v1", credentials=credentials)

        # # Compute Engine service ID is a fixed constant — no need to look it up
        # compute_service = "services/6F81-5844-456A"

        # cpu_price = 0.0
        # ram_price = 0.0
        # page_token = None
        # while True:
        #     kwargs = {"parent": compute_service, "pageSize": 500}
        #     if page_token:
        #         kwargs["pageToken"] = page_token
        #     resp = billing.services().skus().list(**kwargs).execute()

        #     for sku in resp.get("skus", []):
        #         desc = sku["description"]
        #         if region not in sku.get("serviceRegions", []):
        #             continue
        #         if f"Spot Preemptible {family} Instance" not in desc:
        #             continue
        #         tiers = sku["pricingInfo"][0]["pricingExpression"]["tieredRates"]
        #         unit_price = tiers[0]["unitPrice"]
        #         price = float(unit_price["units"]) + unit_price.get("nanos", 0) / 1e9
        #         if "Core" in desc and cpu_price == 0.0:
        #             cpu_price = price
        #         elif "Ram" in desc and ram_price == 0.0:
        #             ram_price = price

        #     page_token = resp.get("nextPageToken")
        #     if not page_token or (cpu_price > 0.0 and ram_price > 0.0):
        #         break

        # if cpu_price == 0.0 and ram_price == 0.0:
        #     raise ValueError(
        #         f"No Spot prices found for {machine_type} (family {family}) in {region}"
        #     )

        # price = vcpus * cpu_price + memory_gb * ram_price
        # print(f"gcp: price {price}")
        # return price
    
    def provider_machine_type(self, node_config: Dict[str, Any]) -> str:
        return node_config["machineType"]
