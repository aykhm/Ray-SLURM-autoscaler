import copy
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

import boto3

from ray.autoscaler._private.aws.node_provider import (
    AWSNodeProvider as RayAwsNodeProvider,
)
from ray.autoscaler._private.slurm.node_provider import SlurmClusterState

logger = logging.getLogger(__name__)


class AwsNodeProvider:
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

        self._delegate = RayAwsNodeProvider(provider_config, cluster_name)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)

        if "aws" in config["provider"]:
            # Fix config for bootstrapping by moving some fields to top level
            config["auth"]["ssh_user"] = config["provider"]["aws"]["auth"]["ssh_user"]
            provider_keys = config["provider"]["aws"].keys()
            for k in provider_keys:
                config["provider"][k] = config["provider"]["aws"][k]

            import ray.autoscaler._private.aws.config as aws_config

            _configure_iam_role = aws_config._configure_iam_role
            aws_config._configure_iam_role = lambda c: (
                c
            )  # monkeypatch the configure_iam_role part
            # of RawAwsNodeProvider.bootstrap_config to
            # be a noop, since it assumes that the head
            # node is under aws.

            # Run bootstrap config
            config = RayAwsNodeProvider.bootstrap_config(config)

            # Undo fixes
            config["auth"].pop("ssh_user")
            for k in provider_keys:
                config["provider"].pop(k)

            aws_config._configure_iam_role = _configure_iam_role

        return config

    @staticmethod
    def fillout_available_node_types_resources(
        cluster_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = RayAwsNodeProvider.fillout_available_node_types_resources(config)
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

            # TODO put this into the config file instead.
            ray_start_command = "ray start"
            ray_start_command += ' --address="localhost:6379"'
            ray_start_command += ' --node-ip-address="' + node_ip + '"'
            ray_start_command += (
                ' --redis-password="' + meta_info["redis_password"] + '"'
            )

            logger.info(f"Run init command ({ray_start_command})\n")
            self.get_command_runner(
                "AwsNodeProvider create:",
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
        tag_filters = tag_filters.copy()  # AWSNodeProvider modifies tag_filters
        return self._delegate.non_terminated_nodes(tag_filters)

    def is_running(self, node_id: str) -> bool:
        return self._delegate.is_running(node_id)

    def is_terminated(self, node_id: str) -> bool:
        return self._delegate.is_terminated(node_id)

    def set_node_tags(self, node_id: str, tags: Dict[str, str]) -> None:
        return self._delegate.set_node_tags(node_id, tags)

    def node_tags(self, node_id: str) -> Dict[str, str]:
        return self._delegate.node_tags(node_id)

    def external_ip(self, node_id: str) -> Optional[str]:
        return self._delegate.external_ip(node_id)

    def internal_ip(self, node_id: str) -> Optional[str]:
        # Force use of external IP since we're using use_internal_ips: False
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

    def get_spot_price(self, node_config: Dict[str, Any]) -> float:
        instance_type = node_config["InstanceType"]

        client = boto3.client("ec2", region_name=self.provider_config["region"])
        response = client.describe_spot_price_history(
            InstanceTypes=[instance_type],
            ProductDescriptions=["Linux/UNIX"],
            AvailabilityZone=self.provider_config["availability_zone"],
            MaxResults=1,
        )
        history = response.get("SpotPriceHistory", [])
        if not history:
            print(f"aws error: No spot price found for {instance_type}")
            return 1e9

        price = float(history[0]["SpotPrice"])
        print(f"aws: price {price}")
        return price
