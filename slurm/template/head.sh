#!/bin/bash -l

head_node_ip="[_PY_HEAD_NODE_IP_]" # To be replaced by python script
port="[_PY_PORT_]"
ray_client_port="[_PY_RAY_CLIENT_PORT_]"
dashboad_port="[_PY_DASHBOARD_PORT_]"
password="[_PY_REDIS_PASSWORD_]"

cpus="[_DEPLOY_HEAD_CPUS_]"

[_PY_INIT_COMMAND_] # To be replaced by python laucher

ray start --head --node-ip-address="$head_node_ip" --port=$port --dashboard-port=$dashboad_port \
    --num-cpus "$cpus" --ray-client-server-port "$ray_client_port" \
    --node-manager-port=7000 --object-manager-port=7001 --runtime-env-agent-port=7002 \
    --min-worker-port=10002 --max-worker-port=10100 \
    --autoscaling-config=~/ray_bootstrap_config.yaml --redis-password="$password"

# TODO: hardcoded port numbers