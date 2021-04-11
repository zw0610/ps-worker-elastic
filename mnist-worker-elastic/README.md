# mnist worker elastic

在本样例中，会进行弹性的 mnist 训练，其中 worker 数量会在不同 epoch 之间变化。主要流程为

## 环境要求

需要下载 tensorflow 2.5.0-rc0，因为 2.4.1 中 `Worker` 还没有加上 `stop` 函数。

结合本目录中的 `cluster_coordinator.py.diff` 修改安装目录下的 `cluster_coordinator.py` 文件。这个修改主要是给 `Worker` 添加了 `restart` 功能，以及修复了原先的一个 bug。其中加的一行 `print` 是用来查看当前的 `Closure` 是被调度在哪个 `Worker` 上了。

## 运行方式

在运行环境中加入环境变量：

```bash
export GRPC_FAIL_FAST=use_caller
```

然后先后运行 `prepare.sh` 和 `run.sh`。（建议在两个 terminal 里面分别跑，注意都需要加环境变量。）

## 效果

一共训练 5 个 epoch。每个 epoch 参与的 worker 为：

- epoch 1: 0, 1, 2 worker
- epoch 2: 0, 1, 2, 3 worker
- epoch 3: 0, 1, 2, 3, 4 worker
- epoch 4: 1, 2, 3, 4 worker
- epoch 5: 0, 1, 2, 3, 4 worker

也就是说，初始的 epoch 1 使用 3 个 worker；epoch 2 和 3 均会增加一个 worker；epoch 4 停掉 worker 0；epoch 5 重启 worker 0。

为了方便，会在 `prepare.sh` 中把所有的 5 个 server 都启动起来。

通过上述 `.diff` 文件中的 log，可以在生成的 `coordinator.log` 文件中看到每次 schedule 到哪个 worker 上。以及在每个 `worker_x.log` 中看到这个 worker 都在哪个 epoch 参与了计算。

5 个 epoch，每个 epoch 50 步，train accuracy 升到 95% 左右。
