import os
import tensorflow as tf


class RoleType:
    coordinator = "coordinator"
    ps = "ps"
    worker = "worker"


NUM_WORKERS = 3
NUM_PS = 2


def get_my_resolver(role: str, idx: int or None) -> tf.distribute.cluster_resolver.ClusterResolver:
    tf_config_str = ""
    if role == RoleType.coordinator:
        tf_config_str = '''
      {
        "cluster":{
          "ps": ["localhost:2001", "localhost:2002"],
          "worker": ["localhost:2101", "localhost:2102",
                     "localhost:2103"]
        }
      }
      '''
    else:
        tf_config_str_template = '''
      {
        "cluster":{
          "ps": ["localhost:2001", "localhost:2002"],
          "worker": ["localhost:2101", "localhost:2102",
                     "localhost:2103"]
        },
        "task": {
          "type": "{role}",
          "index": {idx}
        }
      }
      '''
        tf_config_str = tf_config_str_template.replace("{role}", role).replace("{idx}", str(idx))

    os.environ['TF_CONFIG'] = tf_config_str

    return tf.distribute.cluster_resolver.TFConfigClusterResolver()
