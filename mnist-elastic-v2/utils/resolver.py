import os
import tensorflow as tf
from tensorflow.python.training import server_lib
from pprint import pprint

from kubernetes import config as k8sconfig
from kubernetes import client as k8sclient


default_port = 2222


class TFJobResolver(tf.distribute.cluster_resolver.KubernetesClusterResolver):
    group_name_key = "group-name"
    group_name_val = "kubeflow.org"
    replica_type_key = "replica-type"
    replica_type_worker = "worker"
    replica_type_ps = "ps"
    job_name_key = "job-name"

    def __init__(self, tf_job_name: str, server_port=default_port):
        kubeflow_label = f"{TFJobResolver.group_name_key}={TFJobResolver.group_name_val}"
        worker_label = f"{TFJobResolver.replica_type_key}={TFJobResolver.replica_type_worker}"
        ps_label = f"{TFJobResolver.replica_type_key}={TFJobResolver.replica_type_ps}"
        job_name_label = f"{TFJobResolver.job_name_key}={tf_job_name}"
        tf_job_to_label_mapping = {
            TFJobResolver.replica_type_worker: [",".join([kubeflow_label, worker_label, job_name_label])],
            TFJobResolver.replica_type_ps: [",".join([kubeflow_label, ps_label, job_name_label])]
        }
        print("tf_job_to_label_mapping:")
        pprint(tf_job_to_label_mapping)

        k8sconfig.load_kube_config()
        self._k8s_client = k8sclient.CoreV1Api()

        super().__init__(
            job_to_label_mapping=tf_job_to_label_mapping,
            tf_server_port=server_port,
            override_client=self._k8s_client
        )

    def cluster_spec(self):
        """We generally copy what the parent class does, but replace the
        raise RuntimeError with print as we are doing something elastic here
        """

        cluster_map = {}

        for replica_type in self._job_to_label_mapping:
            all_pods = []
            assert len(self._job_to_label_mapping[replica_type]) == 1
            for selector in self._job_to_label_mapping[replica_type]:
                ret = self._k8s_client.list_pod_for_all_namespaces(label_selector=selector)
                selected_pods = []

                # Sort the list by the name to make sure it doesn't change call to call.
                for pod in sorted(ret.items, key=lambda x: x.metadata.name):
                    if pod.status.phase == 'Running':
                        selected_pods.append(
                            f"{pod.metadata.name}.{pod.metadata.namespace}.svc.cluster.local:{self._tf_server_port}")
                    else:
                        print('Pod "%s" is not running; phase: "%s"' %
                                           (pod.metadata.name, pod.status.phase))
                all_pods.extend(selected_pods)
            cluster_map[replica_type] = all_pods

        return server_lib.ClusterSpec(cluster_map)
