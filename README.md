# README

使用GCN来做下面任务

- predict the edge ratings *预测边的排名*
  - metric: RMSE
- predict edge existence   *预测边的存在性*
  - metric: Recall@K

## 环境

```bash
Fri Dec 22 10:31:27 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4090        Off | 00000000:D6:00.0 Off |                  Off |
| 34%   34C    P8              17W / 450W |      2MiB / 24564MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## 数据集

- [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) (size: 1 MB)
- [ml-latest.zip](https://files.grouplens.org/datasets/movielens/ml-latest.zip) (size: 335 MB)
