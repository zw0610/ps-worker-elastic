import os
import subprocess
import time
import tensorflow as tf

from utils.resolver import get_coordinator_resolver
from utils.model import MyModel, get_datasets
from tensorflow.python.eager import context

from tensorflow.python.distribute.coordinator.cluster_coordinator import Worker

if __name__ == "__main__":
    cluster_resolver = get_coordinator_resolver(num_workers=2)

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

    def get_step_fn(comment=""):
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
            return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
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

    per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
    per_worker_train_iter = iter(per_worker_train_ds)

    EPOCHS = 9
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        coordinator.join()

        step = 50
        if epoch == 1:
            # add worker 3
            coordinator.add_worker("localhost:2103")
            per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
            per_worker_train_iter = iter(per_worker_train_ds)

        if epoch == 2:
            # add worker 4
            coordinator.add_worker("localhost:2104")
            per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
            per_worker_train_iter = iter(per_worker_train_ds)

        if epoch == 3:
            # stop worker 0
            coordinator.remove_worker("localhost:2101")
            # manually kill worker 0 with hardcoding
            pids = os.popen("ps -a | grep 'port 2101' | awk '{print $1}'").read().split("\n")
            os.system("kill {}".format(" ".join(pids)))
            time.sleep(5)

        if epoch == 4:
            # restart worker 0
            proc = subprocess.Popen('python3 ./worker.py --role=worker --idx=0 --port 2101 >> ./worker_0.log 2>&1',
                                    shell=True)
            # wait for the new server to start
            time.sleep(5)
            coordinator.add_worker("localhost:2101")
            per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
            per_worker_train_iter = iter(per_worker_train_ds)

        if epoch == 5:
            # stop worker 1
            coordinator.remove_worker("localhost:2102")
            # manually kill worker 0 with hardcoding
            pids = os.popen("ps -a | grep 'port 2102' | awk '{print $1}'").read().split("\n")
            os.system("kill {}".format(" ".join(pids)))

        if epoch == 6:
            # add worker 5
            coordinator.add_worker("localhost:2105")
            per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
            per_worker_train_iter = iter(per_worker_train_ds)

        if epoch == 7:
            # stop worker 3
            coordinator.remove_worker("localhost:2103")
            # manually kill worker 0 with hardcoding
            pids = os.popen("ps -a | grep 'port 2103' | awk '{print $1}'").read().split("\n")
            os.system("kill {}".format(" ".join(pids)))

        if epoch == 8:
            # add worker 6
            coordinator.add_worker("localhost:2106")
            per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
            per_worker_train_iter = iter(per_worker_train_ds)

        step_fn = get_step_fn(epoch + 1)
        for i in range(step):
            coordinator.schedule(step_fn,
                                 args=(per_worker_train_iter,))
        coordinator.join()
        train_loss_result = train_loss.result()
        train_accuracy_result = train_accuracy.result()
        coordinator.join()
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss_result}, '
            f'Accuracy: {train_accuracy_result * 100}, ',
            flush=True
        )
    proc.terminate()