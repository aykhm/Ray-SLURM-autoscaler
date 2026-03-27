import copy
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

from ray.autoscaler._private.aws.node_provider import AWSNodeProvider as RayAwsNodeProvider
from ray.autoscaler._private.slurm.slurm_node_provider import SlurmClusterState

logger = logging.getLogger(__name__)

fix_config_keys = ["region", "aws_credentials", "use_internal_ips"]
AWS_PREFIX = "aws:" # TODO
AWS_USER = "ec2-user" # TODO

# TODO hacky
def fix_config_provider(config: Dict[str, Any]):
    for k in fix_config_keys:
        config['provider'][k] = config['provider']['aws'].get(k, {})

def undo_fix_config_provider(config: Dict[str, Any]):
    for k in fix_config_keys:
        config['provider'].pop(k, None)

class AwsNodeProvider:
    def __init__(self, provider_config: Dict[str, Any], cluster_name: str, slurm_cluster_state: SlurmClusterState) -> None:
        self.slurm_cluster_state = slurm_cluster_state
        self.cluster_name = cluster_name
        self._delegate = RayAwsNodeProvider(provider_config, cluster_name)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        fix_config_provider(config)
        config = RayAwsNodeProvider.bootstrap_config(config)
        undo_fix_config_provider(config)
        return config

    @staticmethod
    def fillout_available_node_types_resources(
        cluster_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        fix_config_provider(config)
        config = RayAwsNodeProvider.fillout_available_node_types_resources(
            config
        )

        undo_fix_config_provider(config)
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
                ports.extend(range(10002, 10100)) # TODO hacky, should be configurable

                tunnel_cmd = [
                    "ssh",
                    "-i", os.path.expanduser("~/ray_bootstrap_key.pem"),
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ConnectTimeout=10",
                    "-o", "ExitOnForwardFailure=yes",
                    "-o", "ServerAliveInterval=5",
                    "-o", "ServerAliveCountMax=3",
                    "-f",
                    "-N",
                ]

                for port in ports:
                    tunnel_cmd.append("-R")
                    tunnel_cmd.append(f"{port}:localhost:{port}")

                tunnel_cmd.append(f"{AWS_USER}@{node_ip}")

                try:
                    print(f"Run SSH tunneling command for {node_id}\n")
                    subprocess.run(tunnel_cmd, check=True)
                    print(f"SSH tunnel setup for {node_id}\n")
                    break
                except subprocess.CalledProcessError as e:
                    logger.warning(f"SSH tunneling command failed for {node_id}: " + str(e))
                    time.sleep(10)

            # TODO put this into the config file instead.
            ray_start_command = "ray start"
            ray_start_command += " --address=\"localhost:6379\""
            ray_start_command += " --node-ip-address=\"" + node_ip + "\""
            ray_start_command += " --redis-password=\"" + meta_info["redis_password"] + "\""

            logger.info(f"Run init command ({ray_start_command})\n")
            # TODO fix auth config
            self.get_command_runner("AwsNodeProvider create:", node_id, {"ssh_user": AWS_USER, "ssh_private_key": "~/ray_bootstrap_key.pem"}, self.cluster_name, subprocess, False).run(ray_start_command)

        prefixed_res = {}
        for raw_id, instance in res.items():
            prefixed_res[AWS_PREFIX + raw_id] = instance
        return prefixed_res

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
        tag_filters = tag_filters.copy() # AWSNodeProvider modifies tag_filters
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
        use_internal_ip = False # TODO hacky
        return self._delegate.get_command_runner(
            log_prefix,
            node_id,
            auth_config,
            cluster_name,
            process_runner,
            use_internal_ip,
            docker_config,
        )
