from typing import Dict


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
