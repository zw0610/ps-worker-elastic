import os
import tensorflow as tf

def get_coordinator_resolver(num_workers):
    tf_config_str_template = '''
      {
        "cluster":{
          "ps": ["localhost:2001"],
          "worker": [{workers}],
          "chief": ["localhost:2201"]
        },
        "task": {
          "type": "chief",
          "index": 0
        }
      }
      '''
    
    worker_str = ""
    for i in range(num_workers):
        if i != 0:
            worker_str += ","
        worker_str += '"localhost:{}"'.format(i + 2101)
    tf_config_str = tf_config_str_template.replace("{workers}", worker_str)

    os.environ['TF_CONFIG'] = tf_config_str

    return tf.distribute.cluster_resolver.TFConfigClusterResolver()
