import os
import argparse
import tensorflow as tf

from utils.resolver import get_my_resolver
from utils.resolver import RoleType

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role',
                        type=str,
                        required=True,
                        choices=[RoleType.ps, RoleType.worker],
                        help='role, ps or worker.')

    parser.add_argument('--idx',
                        type=int,
                        required=True,
                        help='task index of the role.')

    args = parser.parse_args()

    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    # Provide a `tf.distribute.cluster_resolver.ClusterResolver` that serves
    # the cluster information. See below "Cluster setup" section.
    cluster_resolver = get_my_resolver(role=args.role, idx=args.idx)

    server = tf.distribute.Server(
        cluster_resolver.cluster_spec(),
        job_name=cluster_resolver.task_type,
        task_index=cluster_resolver.task_id,
        protocol="grpc")

    # Blocking the process that starts a server from exiting.
    server.join()
