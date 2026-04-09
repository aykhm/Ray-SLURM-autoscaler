import copy
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from ray.autoscaler._private.multicloud.aws_node_provider import AwsNodeProvider
from ray.autoscaler._private.multicloud.gcp_node_provider import GcpNodeProvider
from ray.autoscaler._private.slurm.node_provider import SlurmNodeProvider

from ray.autoscaler.tags import (
    TAG_RAY_NODE_STATUS,
    TAG_RAY_USER_NODE_TYPE,
    STATUS_UP_TO_DATE,
)


AWS_PROVIDER = "aws"
GCP_PROVIDER = "gcp"
SLURM_PROVIDER = "slurm"
CLOUD_PROVIDERS = [AWS_PROVIDER, GCP_PROVIDER]
ALL_PROVIDERS = CLOUD_PROVIDERS + [SLURM_PROVIDER]

CHEAPEST_PROVIDER = "cheapest"

BUDGET_UPDATE_INTERVAL_S = 5 * 60  # 5 minutes

TAG_PROVIDER_MACHINE_TYPE = "provider-machine-type"


class CloudBudgeter:
    def __init__(self, state_file: str) -> None:
        self._state_file = state_file

        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            print("Loaded cloud budgeter state", state)
            self._max_spend = state["max_spend"]
            self._spent = state["spent"]
            self._last_update_time = state["last_update_time"]
        else:
            self._max_spend: Dict[str, float] = {}
            self._spent: Dict[str, float] = {}
            self._last_update_time: Dict[str, float] = {}

    def new(self, node_type: str, max_spend: float):
        if node_type in self._max_spend:
            assert self._max_spend[node_type] == max_spend
        else:
            self._max_spend[node_type] = max_spend
            self._spent[node_type] = 0
            self._last_update_time[node_type] = 0
        self._save()

    def update(self, node_type: str, total_rate: float):
        now = time.time()
        if self._last_update_time[node_type] != 0:
            dt_hours = (now - self._last_update_time[node_type]) / 3600
            self._spent[node_type] += total_rate * dt_hours
        self._last_update_time[node_type] = now
        self._save()

    def is_over_budget(self, node_type: str):
        return (
            node_type in self._max_spend
            and self._max_spend[node_type] > 0
            and self._spent[node_type] >= self._max_spend[node_type]
        )

    def _save(self):
        print("Saving cloud budgeter state")
        with open(self._state_file, "w") as f:
            json.dump(
                {
                    "max_spend": self._max_spend,
                    "spent": self._spent,
                    "last_update_time": self._last_update_time,
                },
                f,
            )


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

        # TODO need to delete at end
        state_file = os.path.expanduser(f"~/.ray/budget_{cluster_name}.json")
        self._cloud_budgeter = CloudBudgeter(state_file)

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

        # Start tracking this node type (although it isn't always the first time `create_node` was
        # called for a node type).
        node_type = tags[TAG_RAY_USER_NODE_TYPE]
        if self._cloud_budgeter.is_over_budget(node_type):
            print(f"Not creating a {node_type} (over budget)")
            return None
        self._cloud_budgeter.new(node_type, node_config.get("max_spend", -1))

        node_config = copy.deepcopy(node_config)
        if provider == CHEAPEST_PROVIDER:
            rates = {
                p: self._providers[p].get_spot_rate(
                    self._providers[p].provider_machine_type(node_config[p])
                )
                for p in CLOUD_PROVIDERS
            }
            provider = min(rates.keys(), key=lambda p: rates[p])
            print("rates:", rates, "cheapest provider:", provider)
            node_config = node_config[provider]

        # AWSNodeProvider will complain about unknown 'provider' field.
        if provider == AWS_PROVIDER:
            node_config.pop("provider", None)

        node_ids_and_instances = self._providers[provider].create_node(
            node_config, tags, count
        )

        # Attach provider prefix to node id's.
        prefixed_node_ids_and_instances = {}
        if node_ids_and_instances:
            for raw_id, instance in node_ids_and_instances.items():
                prefixed_node_ids_and_instances[self._prefix(provider, raw_id)] = (
                    instance
                )

        # Add provider specific machine type to tag to be used in post_process.
        for node_id in prefixed_node_ids_and_instances.keys():
            tags = {
                TAG_PROVIDER_MACHINE_TYPE: self._providers[provider].provider_machine_type(node_config)
            }
            self.set_node_tags(node_id, tags)

        return prefixed_node_ids_and_instances

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

        # Create a mapping of worker node type to a list of (node_id, provider_machine_type).
        worker_node_compositions: Dict[str, List[Tuple[str, str]]] = {}
        for provider in CLOUD_PROVIDERS:
            node_ids = [
                self._prefix(provider, id)
                for id in self._providers[provider].non_terminated_nodes({})
            ]
            for node_id in node_ids:
                tags = self.node_tags(node_id)

                # Only count time node is actually running.
                if (
                    tags[TAG_RAY_NODE_STATUS] != STATUS_UP_TO_DATE
                    or TAG_PROVIDER_MACHINE_TYPE not in tags
                ):
                    continue

                node_type = tags[TAG_RAY_USER_NODE_TYPE]
                provider_machine_type = tags[TAG_PROVIDER_MACHINE_TYPE]

                machine = (node_id, provider_machine_type)

                if node_type not in worker_node_compositions:
                    worker_node_compositions[node_type] = []

                worker_node_compositions[node_type].append(machine)

        print("comp", worker_node_compositions)

        # Update the current spent cost for each worker type.
        for node_type, machines in worker_node_compositions.items():
            print("Tracking", node_type)
            if machines:
                # Add the rates of all cloud machines running for this worker type.
                total_rate = 0
                for node_id, machine_type in machines:
                    provider, _ = self._route(node_id)
                    rate = self._providers[provider].get_spot_rate(machine_type)
                    print(f"({provider}), ({machine_type}), ({rate})")
                    total_rate += rate

                self._cloud_budgeter.update(node_type, total_rate)

                if self._cloud_budgeter.is_over_budget(node_type):
                    self.terminate_nodes([node_id for node_id, _ in machines])

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
