# ps-worker-elastic

To support elastic training under TensorFlow Ps-Worker-Strategy.


##  针对 TensorFlow 的修改

## 利用 TF-Job 向 Kubernetes 集群提交作业

### 准备

#### 准备 kubectl

https://kubernetes.io/docs/tasks/tools/

#### 准备 k8s config

有两种方法配置 k8s config：
1. 如果是常用的 config，可以在 $HOME 目录下新建一个 `.kube` 目录，然后将 config 文件重命名为 `config` 文件放在 `.kube` 目录下
2. 如果是临时的 config，可以设置环境变量 `KUBECONFIG`：`export KUBECONFIG=<path_to_k8s_config_file>`

### 提交任务

`kubectl create -f ./distributed_psworker_tfjob.yaml`

#### 查看任务

`kubectl get tfjob`

#### 查看 worker、ps、chief pod

`kubectl get pod`

### 修改 worker 数量

#### 修改 distributed_psworker_tfjob.yaml

1. 编辑 distributed_psworker_tfjob.yaml 文件，讲 `spec.tfReplicaSpecs[Worker].replicas` 从 3 改为其他数量
2. `kubectl apply -f ./distributed_psworker_tfjob.yaml`

按照上述查看 pod 的方法查看 worker pod 是否变化。

*请注意，只有部署了经过修改后的 tf-operator 才支持动态修改，镜像为：`ccr.ccs.tencentyun.com/lucas/tf-operator:latest`*
