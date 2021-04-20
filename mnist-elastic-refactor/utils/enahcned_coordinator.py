from copy import deepcopy
from typing import Dict, List, Any

import tensorflow as tf
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute.coordinator.cluster_coordinator import Worker


def is_cluster_spec_dict_equal(old: Dict, new: Dict) -> bool:
    old_replica_type = set(old.keys())
    new_replica_type = set(new.keys())
    if old_replica_type != new_replica_type:
        return False

    for replica_type in new_replica_type:
        old_pods = set(old[replica_type])
        new_pods = set(new[replica_type])
        if old_pods != new_pods:
            return False

    return True


class ClusterCoordinator(
        tf.distribute.experimental.coordinator.ClusterCoordinator):
    def __init__(self, strategy: distribute_lib.Strategy):
        super(ClusterCoordinator, self).__init__(strategy)
        self._cached_cluster_spec_dict = self._cluster_spec_dict()

    def _cluster_spec_dict(self) -> Dict:
        return self._strategy.cluster_resolver.cluster_spec().as_dict()

    def _topology_changes(self, *args, **kwargs) -> bool:
        new_workers = kwargs[
            "new_workers"] if "new_workers" in kwargs else args[0]
        return len(new_workers) > 0

    def _extract_index_from_worker_name(self, worker_name):
        return int(worker_name.split(".")[0].split("-")[-1])

    def need_new_worker_dataset_iter(self) -> bool:
        new_cluster_spec_dict = self._cluster_spec_dict()
        new_dataset_needed = False

        if is_cluster_spec_dict_equal(self._cached_cluster_spec_dict,
                                      new_cluster_spec_dict):

            new_replicas: Dict[str, List[str]] = dict()
            gone_replicas: Dict[str, List[str]] = dict()

            new_replicas["worker"] = [
                w for w in new_cluster_spec_dict["worker"]
                if w not in self._cached_cluster_spec_dict["worker"]
            ]
            gone_replicas["worker"] = [
                w for w in self._cached_cluster_spec_dict["worker"]
                if w not in new_cluster_spec_dict["worker"]
            ]

            if self._topology_changes(new_workers=new_replicas["worker"]):
                new_dataset_needed = True
                new_cluster_spec_dict_copy = deepcopy(new_cluster_spec_dict)
                new_cluster_spec = tf.train.ClusterSpec(
                    new_cluster_spec_dict_copy)
                print(new_cluster_spec, flush=True)
                tf.config.experimental_connect_to_cluster(
                    cluster_spec_or_resolver=new_cluster_spec,
                    job_name="chief")

            for gw in gone_replicas["worker"]:
                index = self._extract_index_from_worker_name(gw)
                print(f"stopping Worker {index}", flush=True)
                self._cluster.workers[index].stop()

            for nw in new_replicas["worker"]:
                index = self._extract_index_from_worker_name(nw)
                if index < len(self._cluster.workers):
                    print(f"restarting Worker {index}", flush=True)
                    self._cluster.workers[index].restart()
                else:
                    print(f"adding Worker {index}", flush=True)
                    self._cluster.workers.append(
                        Worker(index, f"/job:worker/replica:0/task:{index}",
                               self._cluster))

        self._cached_cluster_spec_dict = new_cluster_spec_dict

        return new_dataset_needed
