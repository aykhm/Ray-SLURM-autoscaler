import copy
from typing import Any, Dict, List, Optional, Tuple

from ray.autoscaler._private.slurm.aws_node_provider import AwsNodeProvider
from ray.autoscaler._private.slurm.gcp_node_provider import GcpNodeProvider
from ray.autoscaler._private.slurm.slurm_node_provider import SlurmNodeProvider

AWS_PREFIX = "aws"
GCP_PREFIX = "gcp"
SLURM_PREFIX = "slurm"
ALL_PREFIXES = [AWS_PREFIX, GCP_PREFIX, SLURM_PREFIX]


class HybridNodeProvider:
    """Hybrid provider that routes calls to Slurm or AWS based on node_config."""

    def __init__(self, provider_config: Dict[str, Any], cluster_name: str) -> None:
        slurm_provider = SlurmNodeProvider(provider_config[SLURM_PREFIX], cluster_name)
        self._providers = {
            AWS_PREFIX: AwsNodeProvider(
                provider_config[AWS_PREFIX], cluster_name, slurm_provider.state
            ),
            GCP_PREFIX: GcpNodeProvider(
                provider_config[GCP_PREFIX], cluster_name, slurm_provider.state
            ),
            SLURM_PREFIX: slurm_provider,
        }

    @staticmethod
    def _node_types_for_provider(cluster_config: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {
            k: v for k, v in cluster_config.get("available_node_types", {}).items()
            if v.get("node_config", {}).get("provider") == prefix
        }

    @staticmethod
    def _bootstrap_scoped(
        cluster_config: Dict[str, Any],
        prefix: str,
        bootstrap_fn,
    ) -> Dict[str, Any]:
        scoped_node_types = HybridNodeProvider._node_types_for_provider(cluster_config, prefix)

        scoped_config = copy.deepcopy(cluster_config)
        scoped_config["available_node_types"] = scoped_node_types
        bootstrapped = bootstrap_fn(scoped_config)

        config = copy.deepcopy(cluster_config)
        for k, v in bootstrapped["available_node_types"].items():
            config["available_node_types"][k] = v

        # Copy bootstrapped auth configs to provider specific section
        config["provider"][prefix]["auth"].update(bootstrapped["auth"])

        return config

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = HybridNodeProvider._bootstrap_scoped(config, AWS_PREFIX, AwsNodeProvider.bootstrap_config)
        config = HybridNodeProvider._bootstrap_scoped(config, GCP_PREFIX, GcpNodeProvider.bootstrap_config)
        config = SlurmNodeProvider.bootstrap_config(config)
        return config

    @staticmethod
    def fillout_available_node_types_resources(
        cluster_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = HybridNodeProvider._bootstrap_scoped(config, AWS_PREFIX, AwsNodeProvider.fillout_available_node_types_resources)
        config = HybridNodeProvider._bootstrap_scoped(config, GCP_PREFIX, GcpNodeProvider.fillout_available_node_types_resources)
        config = SlurmNodeProvider.fillout_available_node_types_resources(config)
        return config

    def prepare_for_head_node(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = self._providers[SLURM_PREFIX].prepare_for_head_node(config)
        return config

    def _route(self, node_id: str) -> Tuple[str, str]:
        for prefix in ALL_PREFIXES:
            if node_id.startswith(prefix):
                return prefix, node_id[len(prefix)+1 :]
        raise ValueError(f"Unknown node_id prefix in {node_id}")

    def _prefix(self, provider: str, node_id: str) -> str:
        for prefix in ALL_PREFIXES:
            if provider == prefix:
                return prefix + ":" + node_id
        raise ValueError(f"Unknown provider {provider}")

    @property
    def max_terminate_nodes(self) -> Optional[int]:
        max = 0
        for prefix in ALL_PREFIXES:
            provider_max = self._providers[prefix].max_terminate_nodes
            if provider_max is None:
                return None
            max += provider_max
        return max

    def is_readonly(self) -> bool:
        for prefix in ALL_PREFIXES:
            if not self._providers[prefix].is_readonly():
                return False

        return True

    def create_node(
        self, node_config: Dict[str, Any], tags: Dict[str, str], count: int
    ) -> Optional[Dict[str, Any]]:
        provider = node_config.get("provider")
        if provider is None:
            raise ValueError("Node config must specify 'provider' field.")

        # AWSNodeProvider will complain about unknown 'provider' field.
        config = copy.deepcopy(node_config)
        config.pop("provider", None)

        res = self._providers[provider].create_node(config, tags, count)

        prefixed_res = {}
        if res:
            for raw_id, instance in res.items():
                prefixed_res[self._prefix(provider, raw_id)] = instance
        return prefixed_res

    def create_node_with_resources_and_labels(
        self,
        node_config: Dict[str, Any],
        tags: Dict[str, str],
        count: int,
        resources: Dict[str, float],
        labels: Dict[str, str],
    ) -> Optional[Dict[str, Any]]:
        return self.create_node(node_config, tags, count)

    def terminate_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        provider, raw_id = self._route(node_id)
        self._providers[provider].terminate_node(raw_id)

    def terminate_nodes(self, node_ids: List[str]) -> Optional[Dict[str, Any]]:
        ids = {}
        for nid in node_ids:
            provider, raw_id = self._route(nid)
            ids.setdefault(provider, []).append(raw_id)

        for provider, raw_ids in ids.items():
            if raw_ids:
                self._providers[provider].terminate_nodes(raw_ids)
        return None

    def non_terminated_nodes(self, tag_filters: Dict[str, str]) -> List[str]:
        nodes: List[str] = []

        for prefix in ALL_PREFIXES:
            for nid in self._providers[prefix].non_terminated_nodes(tag_filters):
                nodes.append(self._prefix(prefix, nid))
        
        return nodes

    def is_running(self, node_id: str) -> bool:
        provider, raw_id = self._route(node_id)
        return self._providers[provider].is_running(raw_id)

    def is_terminated(self, node_id: str) -> bool:
        provider, raw_id = self._route(node_id)
        return self._providers[provider].is_terminated(raw_id)

    def set_node_tags(self, node_id: str, tags: Dict[str, str]) -> None:
        provider, raw_id = self._route(node_id)
        return self._providers[provider].set_node_tags(raw_id, tags)

    def node_tags(self, node_id: str) -> Dict[str, str]:
        provider, raw_id = self._route(node_id)
        return self._providers[provider].node_tags(raw_id)

    def external_ip(self, node_id: str) -> Optional[str]:
        provider, raw_id = self._route(node_id)
        return self._providers[provider].external_ip(raw_id)

    def internal_ip(self, node_id: str) -> Optional[str]:
        provider, raw_id = self._route(node_id)
        return self._providers[provider].internal_ip(raw_id)

    def get_node_id(self, ip_address: str, use_internal_ip: bool = True) -> str:
        for prefix in ALL_PREFIXES:
            try:
                return self._prefix(
                    prefix,
                    self._providers[prefix].get_node_id(ip_address, use_internal_ip),
                )
            except Exception:
                pass
        raise ValueError(f"Node with IP {ip_address} not found in any provider.")

    def safe_to_scale(self) -> bool:
        for prefix in ALL_PREFIXES:
            if not self._providers[prefix].safe_to_scale():
                return False
        return True

    def post_process(self) -> None:
        for prefix in ALL_PREFIXES:
            self._providers[prefix].post_process()

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
        provider, raw_id = self._route(node_id)
        return self._providers[provider].get_command_runner(
            log_prefix,
            raw_id,
            auth_config,
            cluster_name,
            process_runner,
            use_internal_ip,
            docker_config,
        )
