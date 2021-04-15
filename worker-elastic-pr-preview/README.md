# Add elastic training for workers in PSv2

We are glad to have PSv2 last year. The single-client distributed training model is much simpler to use and more intuitive to reason able. We especially love the fault tolerance capability of the new strategy, as most industrial scenarios of the asynchronous training with parameter server usually involve dozens of workers and it's crucial to make sure the job keeps running.

In this PR, we will push the fault tolerance capability a step further -- we are adding elastic training for workers in PSv2, which means we could change the number of the workers during training. This would allow us to adjust the worker size by need and make better usage of the computational resources.

For this purpose, this PR mainly add 2 methods to the `CoordinatorCluster`:

- `remove_worker(self, address)`: Remove a `Worker` from `coordinator._cluster` by its address.
- `add_worker(self, address)`: Add a new `Worker` to `coordinator._cluster` by its address.

Notice that we are leaving the start and stop of the worker servers to the plaform, for instance, KubeFlow.

Before discussing about the underlying mechanism, we'd like to split all `Worker`s in the `coordinator._cluster` to 2 states: running and stopped.

- A running `Worker` is one that keeps on grabbing `Closure`s from `_closure_queue` to process. In other words, a `Worker` is running if its processing thread is running. A typical running `Worker` is one that is just initialized.

- A stopped `Worker` is one that stops getting `Closure`s and whose processing thread has stopped (or `join`ed).

We could call `worker.stop()` to turn a running `Worker` to stopped. And to make a stopped `Worker` back to running, we added a `restart` method to `Worker` class. In `restart` method, the `Worker` will recreate a processing thread if it is stopped.

```
                     ----- stop ----->
__init__ ->  running                   stopped
                     <--- restart ---- 
```

Let's come back to the mechanism of elastic training. In fact, we are just using the `connect_to_cluster` function. This great function could recreate the topology of the server in runtime. But because every `connect_to_cluster` will clean all server caches, which introduce overhead, instead of calling `connect_to_cluster` in every `remove_worker` and `add_worker`,

- in `remove_worker`, we will only stop the `Worker` (by running `worker.stop()`) and add the stopped worker to a dict in `coordinator` called `_stopped_workers`, which is a map from the address of the stopped worker to its index;

- in `add_worker`, if the address is in the `_stopped_workers`, restart the worker to running state (by running `worker.restart()`). Otherwise, we need to use `connect_to_cluster` to recreate the cluster. In this way, we will first remove the stopped workers from the cluster spec and clean the `_stopped_workers`. Then, add the new worker to the cluster spec and call `connect_to_cluster`. Finally, create a new `Worker` and append it to `coordinator._cluster`.

We add some auxiliary functions as well, for example, `add_task` and `remove_task` in `ClusterSpec`. And to support sparse worker_index (worker indices like [0, 2, 5]), we changed the `PerWorkerValues` to hold a dict.

As for the dataset part, we choose to create new iterator after `add_task`. This may be oversimplied. But the dataset part seems to be relatively isolated with the coordinator and can be updated in maybe another PR.

A typical use case of the new apis is:

```python
...
per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
per_worker_train_iter = iter(per_worker_train_ds)

for epoch in range(EPOCHS):
    ...
    if epoch == 0:
        coordinator.remove_worker("localhost:2101")

    if epoch == 1:
        coordinator.add_worker("localhost:2102")
        per_worker_train_ds = coordinator.create_per_worker_dataset(per_worker_dataset_fn)
        per_worker_train_iter = iter(per_worker_train_ds)

    for i in range(step):
        coordinator.schedule(step_fn,
                              args=(per_worker_train_iter,))
    coordinator.join()
    ...
```

Any discussion on the design or any details of this PR is welcomed :). It will be great if you can give us some suggestion on the limitation of this design.

As for the test, we have tested this PR in multiple elastic scenarios. And we hope to discuss with you on how to integrate test for the elastic training as well.

Thank you for your time on reviewing this PR!

Gently ping @yuefengz @rchao.

cc @zw0610
