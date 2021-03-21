## Exp. 1, Binary Version of GAT, SGC, and FastGCN

 

### 1. GAT & Bi-GAT

```sh
python bi-gat.py --gpu 4 --dataset Cora --lr 0.001 --epochs 1500 --early_stopping 0 --exp_name bigat-cora --dropout 0.3 --binarized
python bi-gat.py --gpu 9 --dataset PubMed --lr 0.001 --epochs 1000 --early_stopping 0 --exp_name bigcn --dropout 0.4 --binarized
```

|                       | Cora                    | PubMed                   |
| --------------------- | ----------------------- | ------------------------ |
| **GAT**(Paper Report) | 83.0 ± 0.7              | 79.0 ± 0.3               |
| **Bi-GAT**            | 80.8 ± 0.9              | 77.5 ± 1.0 (**with bn**) |
|                       | 81.1 ± 0.8 (affined-bn) |                          |

### 2. SGC & Bi-SGC

```sh
python sgc.py --binarized --gpu 0 --runs 100 --lr 0.001 --weight_decay 0 --early_stopping 0 --dataset Cora --exp_name bisgc-cora
python sgc.py --binarized --gpu 0 --runs 100 --lr 0.001 --weight_decay 0 --early_stopping 0 --dataset PubMed --exp_name bisgc-pubmed
```


|                       | Cora       | PubMed     |
| --------------------- | ---------- | ---------- |
| **SGC**(Paper Report) | 81.0 ± 0.0 | 78.9 ± 0.0 |
| **Bi-SGC**            | 72.0 ± 1.6 | 74.8 ± 1.4 |


### 3. FastGCN & Bi-FastGCN

```sh
python train-transductive.py --dataset cora --lr 0.01 --epochs 200 --runs 10

python train-transductive.py --dataset cora --lr 0.001 --epochs 200 --runs 10 --dropout 0.4 --binarized


python train-transductive.py --dataset pubmed --lr 0.01 --epochs 200 --runs 10

python train-transductive.py --dataset pubmed --lr 0.001 --epochs 200 --runs 10 --dropout 0.4 --binarized


```



|                                                              | **Cora**   | **PubMed** |
| ------------------------------------------------------------ | ---------- | ---------- |
| ==**FastGCN** (hidden neurons: **16**, publicly released code)== | ==79.8==   | ==78.0==   |
| **FastGCN** (hidden neurons: **64**, publicly released code) | 69.6       | 77.8       |
| **FastGCN** (hidden neurons: **64**, other's pytorch implement) | 79.3 ± 0.8 | 79.1± 0.2  |
| **Bi-FastGCN** (hidden neurons: **64**, pytorch implement)   | 77.0 ± 1.0 | 78.7 ± 0.8 |

## Exp.2, Five Vision Datasets

### GLCN's four vision dataset

文章给出了GCN在MNist等四个图像分类问题的实验结果，但是复现比较困难，原因如下：

无源码；无GCN的构图方法（可能是knn，但如果用knn构图的话，也不知道具体的构图参数）。





### ModelNet40

相关模型是DGCNN：

79.30 (91.05)

88.29

92.89