import os

import argparse
import tensorflow as tf

from tensorflow.core.protobuf.cluster_pb2 import JobDef

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role',
                        type=str,
                        required=True,
                        help='role, ps or worker.')

    parser.add_argument('--idx',
                        type=int,
                        required=True,
                        help='task index of the role.')

    parser.add_argument('--port',
                        type=int,
                        required=True,
                        help='port of the server')

    args = parser.parse_args()

    os.environ["GRPC_FAIL_FAST"] = "use_caller"

    cluster_def = tf.train.ClusterDef()
    job = JobDef()
    job.name = args.role
    job.tasks[args.idx] = "localhost:{}".format(args.port)
    cluster_def.job.append(job)
    print(cluster_def)

    server = tf.distribute.Server(
        cluster_def,
        job_name=args.role,
        task_index=args.idx,
        protocol="grpc")

    # Blocking the process that starts a server from exiting.
    server.join()
