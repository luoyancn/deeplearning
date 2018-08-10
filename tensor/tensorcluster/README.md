# 使用方法

Tensorflow的分布式训练需要启动集群，然后再进行训练任务的启动。

## 启动集群

Tensorflow的集群分为2种角色：worker和ps worker。Worker负责真正的计算，PS worker负责参数处理和同步。

### 启动worker

启动的命令如下

```bash
python grpc_tensorflow_server.py --cluster_spec='worker|localhost:2221;localhost:2222,ps|localhost:2223' --job_name=worker --task_id=0

python grpc_tensorflow_server.py --cluster_spec='worker|localhost:2221;localhost:2222,ps|localhost:2223' --job_name=worker --task_id=1
```

### 启动ps worker

```bash
python grpc_tensorflow_server.py --cluster_spec='worker|localhost:2221;localhost:2222,ps|localhost:2223' --job_name=ps --task_id=0
```

## 启动训练

启动命令如下：

```bash
python diss_saver.py --data_dir data --ps localhost:2223 --ps_index 0 --worker_index 0 --model_save_path models --model_name mnist
```

## 模型推测

启动命令如下：

```bash
python diss_predict.py --model_dir models --ps localhost:2223 --worker_index 0 --ps_index 0 --image 2.png
```

### 参数说明

上述运行集群的命令当中，`cluster_spec`指向的是本地，假设生产环境如下：

- worker
  - tf19-dis-worker-0.tf19-dis-worker:2222
  - tf19-dis-worker-1.tf19-dis-worker:2222

- ps
  - tf19-dis-ps-0.tf19-dis-ps:2222

则上述命令修改如下：

```bash
# 在服务器tf19-dis-worker-0.tf19-dis-ps上执行如下命令
python grpc_tensorflow_server.py --cluster_spec='worker|tf19-dis-worker-0.tf19-dis-worker:2222;tf19-dis-worker-1.tf19-dis-worker:2222,ps|tf19-dis-ps-0.tf19-dis-ps:2222' --job_name=worker --task_id=0

# 在服务器tf19-dis-worker-0.tf19-dis-ps上执行如下命令
python grpc_tensorflow_server.py --cluster_spec='worker|tf19-dis-worker-0.tf19-dis-worker:2222;tf19-dis-worker-1.tf19-dis-worker:2222,ps|tf19-dis-ps-0.tf19-dis-ps:2222' --job_name=worker --task_id=1

# 在服务器tf19-dis-ps-0.tf19-dis-ps上执行如下命令
python grpc_tensorflow_server.py --cluster_spec='worker|tf19-dis-worker-0.tf19-dis-worker:2222;tf19-dis-worker-1.tf19-dis-worker:2222,ps|tf19-dis-ps-0.tf19-dis-ps:2222' --job_name=ps --task_id=0
```

在上面运行训练的命令当中，

- ps指向Tensorflow集群当中的ps worker的地址
- worker_index表示使用第几个worker进行训练
- ps_index表示使用第几个ps worker进行参数处理

如果按照生产环境，则可以使用如下的任何一个命令执行

```bash
python diss_saver.py --data_dir data --ps tf19-dis-ps-0.tf19-dis-ps:2222 --ps_index 0 --worker_index 0 --model_save_path models --model_name mnist

python diss_saver.py --data_dir data --ps tf19-dis-ps-0.tf19-dis-ps:2222 --ps_index 0 --worker_index 1 --model_save_path models --model_name mnist
```

请注意ps_index需要和集群构成的ps的顺序保持一致。

同样的，使用训练好的模型进行推测，其命令应当修改如下:

```bash
python diss_predict.py --model_dir models --ps tf19-dis-ps-0.tf19-dis-ps:2222 --worker_index 0 --ps_index 0 --image images/8.png
```

***请根据实际环境进行参数的修改***

例子的运行结果如图：

![result](result.png)