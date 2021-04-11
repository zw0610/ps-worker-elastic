import os
import socket
import json
import tensorflow as tf

from tensorflow.core.protobuf.cluster_pb2 import JobDef

from utils.resolver import default_port

os.environ["GRPC_FAIL_FAST"] = "use_caller"

if __name__ == '__main__':
    tf_config = os.environ["TF_CONFIG"]
    self_info_dict = json.loads(tf_config)

    role = self_info_dict["task"]["type"]
    idx = int(self_info_dict["task"]["index"])
    localhost = socket.gethostbyname(socket.gethostname())

    job = JobDef()
    job.name = role
    job.tasks[idx] = f"{localhost}:{default_port}"

    cluster_def = tf.train.ClusterDef()
    cluster_def.job.append(job)

    print(cluster_def)

    server = tf.distribute.Server(
        cluster_def,
        job_name=role,
        task_index=idx,
        protocol="grpc")

    # Blocking the process that starts a server from exiting.
    server.join()
