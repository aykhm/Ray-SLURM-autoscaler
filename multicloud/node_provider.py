import copy
from typing import Any, Dict, List, Optional, Tuple

from ray.autoscaler._private.multicloud.aws_node_provider import AwsNodeProvider
from ray.autoscaler._private.multicloud.gcp_node_provider import GcpNodeProvider
from ray.autoscaler._private.slurm.node_provider import SlurmNodeProvider

AWS_PROVIDER = "aws"
GCP_PROVIDER = "gcp"
SLURM_PROVIDER = "slurm"
CLOUD_PROVIDERS = [AWS_PROVIDER, GCP_PROVIDER]
ALL_PROVIDERS = CLOUD_PROVIDERS + [SLURM_PROVIDER]

CHEAPEST_PROVIDER = "cheapest"


class MulticloudNodeProvider:
    """Multicloud provider that routes calls to Slurm or AWS based on node_config."""

    def __init__(self, provider_config: Dict[str, Any], cluster_name: str) -> None:
        self._provider_config = provider_config
        slurm_provider = SlurmNodeProvider(
            provider_config[SLURM_PROVIDER], cluster_name
        )
        self._providers = {
            AWS_PROVIDER: AwsNodeProvider(
                provider_config[AWS_PROVIDER], cluster_name, slurm_provider.state
            ),
            GCP_PROVIDER: GcpNodeProvider(
                provider_config[GCP_PROVIDER], cluster_name, slurm_provider.state
            ),
            SLURM_PROVIDER: slurm_provider,
        }

    @staticmethod
    def _bootstrap_scoped(
        cluster_config: Dict[str, Any],
        prefix: str,
        bootstrap_fn,
    ) -> Dict[str, Any]:
        scoped_config = copy.deepcopy(cluster_config)
        scoped_config["available_node_types"] = {}
        for k, v in cluster_config.get("available_node_types", {}).items():
            node_provider = v.get("node_config", {}).get("provider")
            if node_provider == prefix:
                scoped_config["available_node_types"][k] = v
            elif node_provider == CHEAPEST_PROVIDER and prefix in v.get(
                "node_config", {}
            ):
                # Expose the provider-specific sub-config for bootstrapping
                scoped = copy.deepcopy(v)
                scoped["node_config"] = {**v["node_config"][prefix], "provider": prefix}
                scoped_config["available_node_types"][k] = scoped 

        bootstrapped = bootstrap_fn(scoped_config)

        config = copy.deepcopy(cluster_config)
        for k, v in bootstrapped["available_node_types"].items():
            orig_provider = (
                cluster_config["available_node_types"]
                .get(k, {})
                .get("node_config", {})
                .get("provider")
            )
            if orig_provider == CHEAPEST_PROVIDER:
                # Merge bootstrapped sub-config back into the nested cheapest config
                bootstrapped_sub = copy.deepcopy(v["node_config"])
                bootstrapped_sub.pop("provider", None)
                config["available_node_types"][k]["node_config"][prefix] = (
                    bootstrapped_sub
                )
            else:
                config["available_node_types"][k] = v

        # Copy bootstrapped auth configs to provider specific section
        config["provider"][prefix]["auth"].update(bootstrapped["auth"])

        return config

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = MulticloudNodeProvider._bootstrap_scoped(
            config, AWS_PROVIDER, AwsNodeProvider.bootstrap_config
        )
        config = MulticloudNodeProvider._bootstrap_scoped(
            config, GCP_PROVIDER, GcpNodeProvider.bootstrap_config
        )
        config = SlurmNodeProvider.bootstrap_config(config)
        return config

    @staticmethod
    def fillout_available_node_types_resources(
        cluster_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = MulticloudNodeProvider._bootstrap_scoped(
            config, AWS_PROVIDER, AwsNodeProvider.fillout_available_node_types_resources
        )
        config = MulticloudNodeProvider._bootstrap_scoped(
            config, GCP_PROVIDER, GcpNodeProvider.fillout_available_node_types_resources
        )
        config = SlurmNodeProvider.fillout_available_node_types_resources(config)
        return config

    def prepare_for_head_node(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = self._providers[SLURM_PROVIDER].prepare_for_head_node(config)
        return config

    def _route(self, node_id: str) -> Tuple[str, str]:
        for provider in ALL_PROVIDERS:
            if node_id.startswith(provider):
                return provider, node_id[len(provider) + 1 :]
        raise ValueError(f"Unknown node_id prefix in {node_id}")

    def _prefix(self, provider: str, node_id: str) -> str:
        return provider + ":" + node_id

    @property
    def max_terminate_nodes(self) -> Optional[int]:
        max = 0
        for provider in ALL_PROVIDERS:
            provider_max = self._providers[provider].max_terminate_nodes
            if provider_max is None:
                return None
            max += provider_max
        return max

    def is_readonly(self) -> bool:
        for provider in ALL_PROVIDERS:
            if not self._providers[provider].is_readonly():
                return False

        return True

    def create_node(
        self, node_config: Dict[str, Any], tags: Dict[str, str], count: int
    ) -> Optional[Dict[str, Any]]:
        provider = node_config.get("provider")
        if provider is None:
            raise ValueError("Node config must specify 'provider' field.")

        config = copy.deepcopy(node_config)
        if provider == CHEAPEST_PROVIDER:
            prices = {p: self._providers[p].get_spot_price(node_config[p]) for p in CLOUD_PROVIDERS}
            provider = min(prices.keys(), key=lambda p: prices[p])
            print("prices:", prices, "cheapest provider:", provider)
            config = config[provider]

        # AWSNodeProvider will complain about unknown 'provider' field.
        if provider == AWS_PROVIDER:
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

        for provider in ALL_PROVIDERS:
            for nid in self._providers[provider].non_terminated_nodes(tag_filters):
                nodes.append(self._prefix(provider, nid))

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
        for provider in ALL_PROVIDERS:
            try:
                return self._prefix(
                    provider,
                    self._providers[provider].get_node_id(ip_address, use_internal_ip),
                )
            except Exception:
                pass
        raise ValueError(f"Node with IP {ip_address} not found in any provider.")

    def safe_to_scale(self) -> bool:
        for provider in ALL_PROVIDERS:
            if not self._providers[provider].safe_to_scale():
                return False
        return True

    def post_process(self) -> None:
        for provider in ALL_PROVIDERS:
            self._providers[provider].post_process()

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
