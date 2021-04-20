import os
import time
from copy import deepcopy

from utils.resolver import TFJobResolver
from utils.enhanced_coordinator import is_cluster_spec_dict_equal
from utils.model import MyModel, get_datasets

import tensorflow as tf
from tensorflow.python.distribute.coordinator.cluster_coordinator import Worker

os.environ["GRPC_FAIL_FAST"] = "use_caller"

if __name__ == "__main__":
    job_name = os.environ["JOB_NAME"]

    cluster_resolver = TFJobResolver(tf_job_name=job_name, server_port=2222)

    variable_partitioner = (
        tf.distribute.experimental.partitioners.FixedShardsPartitioner(
            num_shards=1))

    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver,
        variable_partitioner=variable_partitioner)

    with strategy.scope():
        # Create an instance of the model
        model = MyModel()

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            from_logits=True)

        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def get_step_fn(comment="", slow_down=0):
        @tf.function
        def step_fn(iterator):
            def train_step(images, labels):
                with tf.GradientTape() as tape:
                    # training=True is only needed if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).
                    predictions = model(images, training=True)
                    loss = loss_object(labels, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                train_loss(loss)
                train_accuracy(labels, predictions)
                tf.print("loss-{}:".format(comment), tf.math.reduce_mean(loss))
                return loss

            images, labels = next(iterator)
            losses = strategy.run(train_step, args=(images, labels))
            res = strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
            if slow_down > 0:
                time.sleep(slow_down)
            return res

        return step_fn


    # Dispatch to Remote
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(strategy)


    # Must create dataset in this function.
    # Could not use a global function.
    def dataset_fn(_):
        train_ds, test_ds = get_datasets()
        return train_ds


    @tf.function
    def per_worker_dataset_fn():
        return strategy.distribute_datasets_from_function(dataset_fn)

    total_epochs = 10
    total_steps = 300

    cached_cluster_spec = cluster_resolver.cluster_spec().as_dict()
    last_epoch_succeeded = True

    per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_train_iter = iter(per_worker_train_ds)

    for epoch in range(total_epochs):
        print(f"start epoch {epoch}",flush=True)
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        coordinator.join()

    
        if not last_epoch_succeeded:
            print("wait for 15 second for pod status updating",flush=True)
            time.sleep(15)

        new_cluster_spec = cluster_resolver.cluster_spec().as_dict()
        if not is_cluster_spec_dict_equal(cached_cluster_spec, new_cluster_spec):
            new_workers = [w for w in new_cluster_spec["worker"] if w not in cached_cluster_spec["worker"]]
            gone_workers = [w for w in cached_cluster_spec["worker"] if w not in new_cluster_spec["worker"]]

            if len(new_workers) > 0:
                new_cluster_copy = deepcopy(new_cluster_spec)
            
                cluster_spec = tf.train.ClusterSpec(
                    new_cluster_copy
                )

                print(new_cluster_copy,flush=True)

                tf.config.experimental_connect_to_cluster(
                    cluster_spec_or_resolver=cluster_spec,
                    job_name="chief")


            for gw in gone_workers:
                index = int(gw.split(".")[0].split("-")[-1])
                print(f"stopping Worker {index}",flush=True)
                coordinator._cluster.workers[index].stop()

            for nw in new_workers:
                index = int(nw.split(".")[0].split("-")[-1])
                if index < len(coordinator._cluster.workers):
                    print(f"restarting Worker {index}",flush=True)
                    coordinator._cluster.workers[index].restart()
                else:
                    print(f"adding Worker {index}",flush=True)
                    coordinator._cluster.workers.append(
                        Worker(index, f"/job:worker/replica:0/task:{index}", coordinator._cluster)
                    )

            if len(new_workers) > 0:
                per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
                per_worker_train_iter = iter(per_worker_train_ds)

        cached_cluster_spec = new_cluster_spec

        step_fn = get_step_fn(f"{epoch}", slow_down=0)
        for i in range(total_steps):
            coordinator.schedule(step_fn,
                                 args=(per_worker_train_iter,))

        try:
            coordinator.join()
        except Exception as e:
            last_epoch_succeeded = False
            print(e,flush=True)
            print(f"because of the error, we are aborting the current epoch {epoch}",flush=True)
        else:
            last_epoch_succeeded = True
            train_loss_result = train_loss.result()
            train_accuracy_result = train_accuracy.result()
            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {train_loss_result}, '
                f'Accuracy: {train_accuracy_result * 100}, ',
                flush=True
            )
