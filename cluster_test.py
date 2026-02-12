from collections import Counter
import socket
import time

import ray

ray.init(address='auto', namespace='default', runtime_env={'working_dir': '.'})


print('''This cluster consists of
    {} nodes in total
    {} CPU resources in total
'''.format(len(ray.nodes()), ray.cluster_resources()['CPU']))

@ray.remote
def f():
#    time.sleep(0.01)
    time.sleep(3)
    # Return IP address.
    return socket.gethostbyname(socket.gethostname())

object_ids = [f.remote() for _ in range(100)]
ip_addresses = ray.get(object_ids)

print('Tasks executed')
for ip_address, num_tasks in Counter(ip_addresses).items():
    print('    {} tasks on {}'.format(num_tasks, ip_address))
