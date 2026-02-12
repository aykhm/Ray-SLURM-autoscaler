import copy
from typing import Any, Dict, List, Optional, Tuple

from ray.autoscaler._private.slurm.slurm_node_provider import SlurmNodeProvider
from ray.autoscaler._private.slurm.aws_node_provider import AwsNodeProvider

AWS_PREFIX = "aws:"
SLURM_PREFIX = "slurm:"

class HybridNodeProvider:
    """Hybrid provider that routes calls to Slurm or AWS based on node_config."""

    def __init__(self, provider_config: Dict[str, Any], cluster_name: str) -> None:
        # Allow nested configs under provider.slurm / provider.aws; otherwise fallback.
        slurm_cfg = provider_config.get("slurm", provider_config)
        aws_cfg = provider_config.get("aws", provider_config)
        self._slurm = SlurmNodeProvider(slurm_cfg, cluster_name)
        self._aws = AwsNodeProvider(aws_cfg, cluster_name, self._slurm.state)

    @staticmethod
    def bootstrap_config(cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = SlurmNodeProvider.bootstrap_config(config)
        config = AwsNodeProvider.bootstrap_config(config)
        return config

    @staticmethod
    def fillout_available_node_types_resources(
        cluster_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        config = copy.deepcopy(cluster_config)
        config = SlurmNodeProvider.fillout_available_node_types_resources(config)
        config = AwsNodeProvider.fillout_available_node_types_resources(config)
        return config

    def prepare_for_head_node(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        return self._slurm.prepare_for_head_node(cluster_config)

    def _route(self, node_id: str) -> Tuple[str, str]:
        if node_id.startswith(AWS_PREFIX):
            return "aws", node_id[len(AWS_PREFIX) :]
        if node_id.startswith(SLURM_PREFIX):
            return "slurm", node_id[len(SLURM_PREFIX) :]
        return "slurm", node_id

    def _prefix(self, provider: str, node_id: str) -> str:
        if provider == "aws":
            return AWS_PREFIX + node_id
        return SLURM_PREFIX + node_id

    @property
    def max_terminate_nodes(self) -> Optional[int]:
        slurm_max = self._slurm.max_terminate_nodes
        aws_max = self._aws.max_terminate_nodes
        # If either is None (unbounded), treat overall as unbounded.
        if slurm_max is None or aws_max is None:
            return None
        return slurm_max + aws_max

    def is_readonly(self) -> bool:
        return self._slurm.is_readonly() and self._aws.is_readonly()

    def create_node(
        self, node_config: Dict[str, Any], tags: Dict[str, str], count: int
    ) -> Optional[Dict[str, Any]]:
        provider = node_config.get("provider")
        if provider is None and node_config.get("head_node") == 1:
            provider = "slurm"  # Head node is always slurm
        if provider is None:
            raise ValueError("Node config must specify 'provider' field.")

        if provider == "aws":
            # AWSNodeProvider will complain about unknown 'provider' field.
            config = copy.deepcopy(node_config)
            config.pop("provider", None)
            return self._aws.create_node(config, tags, count)
        return self._slurm.create_node(node_config, tags, count)

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
        if provider == "aws":
            return self._aws.terminate_node(raw_id)
        return self._slurm.terminate_node(raw_id)

    def terminate_nodes(self, node_ids: List[str]) -> Optional[Dict[str, Any]]:
        aws_ids, slurm_ids = [], []
        for nid in node_ids:
            provider, raw_id = self._route(nid)
            if provider == "aws":
                aws_ids.append(raw_id)
            else:
                slurm_ids.append(raw_id)
        if aws_ids:
            self._aws.terminate_nodes(aws_ids)
        if slurm_ids:
            self._slurm.terminate_nodes(slurm_ids)
        return None

    def non_terminated_nodes(self, tag_filters: Dict[str, str]) -> List[str]:
        nodes: List[str] = []
        for nid in self._slurm.non_terminated_nodes(tag_filters):
            nodes.append(self._prefix("slurm", nid))
        for nid in self._aws.non_terminated_nodes(tag_filters):
            nodes.append(self._prefix("aws", nid))
        return nodes

    def is_running(self, node_id: str) -> bool:
        provider, raw_id = self._route(node_id)
        if provider == "aws":
            return self._aws.is_running(raw_id)
        return self._slurm.is_running(raw_id)

    def is_terminated(self, node_id: str) -> bool:
        provider, raw_id = self._route(node_id)
        if provider == "aws":
            return self._aws.is_terminated(raw_id)
        return self._slurm.is_terminated(raw_id)

    def set_node_tags(self, node_id: str, tags: Dict[str, str]) -> None:
        provider, raw_id = self._route(node_id)
        if provider == "aws":
            return self._aws.set_node_tags(raw_id, tags)
        return self._slurm.set_node_tags(raw_id, tags)

    def node_tags(self, node_id: str) -> Dict[str, str]:
        provider, raw_id = self._route(node_id)
        if provider == "aws":
            return self._aws.node_tags(raw_id)
        return self._slurm.node_tags(raw_id)

    def external_ip(self, node_id: str) -> Optional[str]:
        provider, raw_id = self._route(node_id)
        if provider == "aws":
            return self._aws.external_ip(raw_id)
        return self._slurm.external_ip(raw_id)

    def internal_ip(self, node_id: str) -> Optional[str]:
        provider, raw_id = self._route(node_id)
        if provider == "aws":
            return self._aws.internal_ip(raw_id)
        return self._slurm.internal_ip(raw_id)

    def get_node_id(self, ip_address: str, use_internal_ip: bool = True) -> str:
        # Try slurm first (legacy behavior), then AWS.
        try:
            return self._prefix(
                "slurm", self._slurm.get_node_id(ip_address, use_internal_ip)
            )
        except Exception:
            pass
        return self._prefix("aws", self._aws.get_node_id(ip_address, use_internal_ip))

    def safe_to_scale(self) -> bool:
        return self._slurm.safe_to_scale() and self._aws.safe_to_scale()

    def post_process(self) -> None:
        self._slurm.post_process()
        self._aws.post_process()

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
        if provider == "aws":
            return self._aws.get_command_runner(
                log_prefix,
                raw_id,
                auth_config,
                cluster_name,
                process_runner,
                use_internal_ip,
                docker_config,
            )
        return self._slurm.get_command_runner(
            log_prefix,
            raw_id,
            auth_config,
            cluster_name,
            process_runner,
            use_internal_ip,
            docker_config,
        )
