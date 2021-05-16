## Exp. 1, Binary Version of GAT, SGC, and FastGCN

 

### 1. GAT & Bi-GAT

```sh
python bi-gat.py --gpu 0 --dataset Cora --lr 0.001 --epochs 1500 --early_stopping 0 --exp_name bigat-cora --dropout 0.3 --binarized
python bi-gat.py --gpu 2 --dataset CiteSeer --lr 0.001 --epochs 1000 --early_stopping 0 --exp_name bigat-pubmed --dropout 0.3 --binarized
python bi-gat.py --gpu 2 --dataset PubMed --lr 0.001 --epochs 1000 --early_stopping 0 --exp_name bigat-pubmed --dropout 0.4 --weight_decay 0 --binarized
```

|                         | Cora               | PubMed                   |
| ----------------------- | ------------------ | ------------------------ |
| **GAT**(Paper Report)   | 83.0 ± 0.7         | 79.0 ± 0.3               |
| **Bi-GAT**              | 80.8 ± 0.9         | 77.5 ± 1.0 (**with bn**) |
| **Bi-GAT** (affined-bn) | 81.1 ± 0.8 (False) | 78.4±1.0 (True)          |0.6815, std: 0.0088 (20 runs)|

### 2. SGC & Bi-SGC

```sh
python sgc.py --binarized --gpu 0 --runs 100 --lr 0.001 --weight_decay 0 --early_stopping 0 --dataset Cora --exp_name bisgc-cora
python sgc.py --binarized --gpu 0 --runs 100 --lr 0.001 --weight_decay 0 --early_stopping 0 --dataset PubMed --exp_name bisgc-pubmed
python sgc.py --binarized --gpu 4 --runs 20 --lr 0.001 --weight_decay 0 --early_stopping 0 --dataset CiteSeer --exp_name bisgc-citeseer
t CiteSeer --exp_name bisgc-citeseer
```


|                       | Cora       | PubMed     | Citeseer   |
| --------------------- | ---------- | ---------- | ---------- |
| **SGC**(Paper Report) | 81.0 ± 0.0 | 78.9 ± 0.0 |
| **Bi-SGC**            | 72.0 ± 1.6 | 74.8 ± 1.4 |0.6087, std: 0.0139(20runs)|


### 3. FastGCN & Bi-FastGCN

```sh
python train-transductive.py --dataset cora --lr 0.01 --epochs 200 --runs 10

python train-transductive.py --dataset cora --lr 0.001 --epochs 200 --runs 10 --dropout 0.4 --binarized


python train-transductive.py --dataset pubmed --lr 0.01 --epochs 200 --runs 10

python train-transductive.py --dataset pubmed --lr 0.001 --epochs 200 --runs 10 --dropout 0.1 --binarized

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


### OGBN

|               | Products                          | Proteins                  |
| ------------- | --------------------------------- | ------------------------- |
| GCN           |                                   |                           |
| Bi-GCN        |                                   |                           |
| GraphSAGE     | 78.7 ± 0.4 (inductive)            | 77.62 ± 0.52 (full batch) |
| Bi-GraphSAGE  | 76.8 ± 0.3 (inductive)            | 73.10 ± 0.72 (full batch) |
| GraphSAINT    | 79.1 ± 0.3 (inductive, sage aggr) |                           |
| Bi-GraphSAINT | 77.5 ± 0.4 (inductive, sage aggr) |                           |

```bash
# products-graphsage
python sage-ogb.py --gpu 0 --runs 10
# products-bigraphsage (0.7456 ± 0.0092)
python sage-ogb.py --gpu 1 --binarized
# 0.7447 ± 0.0094
python sage-ogb.py --gpu 4 --binarized --lr 0.001


# products-graphsaint

# products-graphsaint


# proteins-graphsage
python proteins-gnn.py --gpu 2 --use_sage
# proteins-bigraphsage
python proteins-gnn.py --gpu 2 --use_sage --binarized
#
```



## Exp.3, Entropy Cover hypothesis

### Reddit

|      | GCN (inductive)  | Bi-GCN (inductive) |
| ---- | ---------------- | ------------------ |
| 1    | 38.59 ± 3.39     | -                  |
| 2    | 63.63 ± 1.38     |                    |
| 4    | 87.05 ± 0.49     |                    |
| 8    | 92.11 ± 0.14     |                    |
| 16   | 93.31 ± 0.12     | 0                  |
| 32   | ==93.85 ± 0.07== | 0                  |
| 64   | 93.87 ± 0.07     | 0.8298 ± 0.2916    |
| 128  | 93.86 ± 0.09     | 0.9292 ± 0.0015    |
| 256  | 93.78 ± 0.07     | 0.9310 ± 0.0028    |
| 512  | 93.75 ± 0.12     | 0.9307 ± 0.0017    |
| 1024 | 0.9355 ± 0.0013  | 0.9320 ± 0.0018    |

```
log

python ind-gcn.py --gpu 0 --dataset Reddit --hidden 1
Run: 0, best_test: 0.3850
Run: 1, best_test: 0.3859
Run: 2, best_test: 0.3944
Run: 3, best_test: 0.3630
Run: 4, best_test: 0.3445
Run: 5, best_test: 0.4333
Run: 6, best_test: 0.3264
Run: 7, best_test: 0.3906
Run: 8, best_test: 0.4089
Run: 9, best_test: 0.4266
avg test f1 score:0.3859 ± 0.0339
python ind-gcn.py --gpu 9 --dataset Reddit --hidden 2
Run: 0, best_test: 0.6548
Run: 1, best_test: 0.6338
Run: 2, best_test: 0.6240
Run: 3, best_test: 0.6321
Run: 4, best_test: 0.6418
Run: 5, best_test: 0.6358
Run: 6, best_test: 0.6196
Run: 7, best_test: 0.6394
Run: 8, best_test: 0.6611
Run: 9, best_test: 0.6202
avg test f1 score:0.6363 ± 0.0138
python ind-gcn.py --gpu 3 --dataset Reddit --hidden 4
Run: 0, best_test: 0.8767
Run: 1, best_test: 0.8767
Run: 2, best_test: 0.8662
Run: 3, best_test: 0.8657
Run: 4, best_test: 0.8716
Run: 5, best_test: 0.8704
Run: 6, best_test: 0.8696
Run: 7, best_test: 0.8681
Run: 8, best_test: 0.8636
Run: 9, best_test: 0.8767
avg test f1 score:0.8705 ± 0.0049
python ind-gcn.py --gpu 3 --dataset Reddit --hidden 8
Run: 0, best_test: 0.9214
Run: 1, best_test: 0.9218
Run: 2, best_test: 0.9185
Run: 3, best_test: 0.9203
Run: 4, best_test: 0.9216
Run: 5, best_test: 0.9213
Run: 6, best_test: 0.9236
Run: 7, best_test: 0.9200
Run: 8, best_test: 0.9218
Run: 9, best_test: 0.9206
avg test f1 score:0.9211 ± 0.0014
python ind-gcn.py --gpu 3 --dataset Reddit --hidden 16
Run: 0, best_test: 0.9316
Run: 1, best_test: 0.9345
Run: 2, best_test: 0.9337
Run: 3, best_test: 0.9336
Run: 4, best_test: 0.9324
Run: 5, best_test: 0.9332
Run: 6, best_test: 0.9336
Run: 7, best_test: 0.9324
Run: 8, best_test: 0.9313
Run: 9, best_test: 0.9350
avg test f1 score:0.9331 ± 0.0012
python ind-gcn.py --gpu 9 --dataset Reddit --hidden 32
Run: 0, best_test: 0.9385
Run: 1, best_test: 0.9377
Run: 2, best_test: 0.9385
Run: 3, best_test: 0.9393
Run: 4, best_test: 0.9389
Run: 5, best_test: 0.9380
Run: 6, best_test: 0.9399
Run: 7, best_test: 0.9383
Run: 8, best_test: 0.9377
Run: 9, best_test: 0.9380
avg test f1 score:0.9385 ± 0.0007
python ind-gcn.py --gpu 3 --dataset Reddit --hidden 64
Run: 0, best_test: 0.9395
Run: 1, best_test: 0.9376
Run: 2, best_test: 0.9398
Run: 3, best_test: 0.9385
Run: 4, best_test: 0.9386
Run: 5, best_test: 0.9395
Run: 6, best_test: 0.9392
Run: 7, best_test: 0.9382
Run: 8, best_test: 0.9380
Run: 9, best_test: 0.9384
avg test f1 score:0.9387 ± 0.0007
python ind-gcn.py --gpu 1 --dataset Reddit --hidden 128
Run: 0, best_test: 0.9375
Run: 1, best_test: 0.9390
Run: 2, best_test: 0.9377
Run: 3, best_test: 0.9383
Run: 4, best_test: 0.9392
Run: 5, best_test: 0.9378
Run: 6, best_test: 0.9387
Run: 7, best_test: 0.9399
Run: 8, best_test: 0.9400
Run: 9, best_test: 0.9379
avg test f1 score:0.9386 ± 0.0009
python ind-gcn.py --gpu 0 --dataset Reddit --hidden 256
Run: 0, best_test: 0.9377
Run: 1, best_test: 0.9376
Run: 2, best_test: 0.9381
Run: 3, best_test: 0.9387
Run: 4, best_test: 0.9379
Run: 5, best_test: 0.9388
Run: 6, best_test: 0.9380
Run: 7, best_test: 0.9365
Run: 8, best_test: 0.9375
Run: 9, best_test: 0.9376
avg test f1 score:0.9378 ± 0.0007
python ind-gcn.py --gpu 1 --dataset Reddit --hidden 512
Run: 0, best_test: 0.9385
Run: 1, best_test: 0.9382
Run: 2, best_test: 0.9379
Run: 3, best_test: 0.9353
Run: 4, best_test: 0.9375
Run: 5, best_test: 0.9384
Run: 6, best_test: 0.9375
Run: 7, best_test: 0.9380
Run: 8, best_test: 0.9386
Run: 9, best_test: 0.9354
avg test f1 score:0.9375 ± 0.0012
python ind-gcn.py --gpu 1 --dataset Reddit --hidden 1024
Run: 0, best_test: 0.9354
Run: 1, best_test: 0.9336
Run: 2, best_test: 0.9367
Run: 3, best_test: 0.9368
Run: 4, best_test: 0.9328
Run: 5, best_test: 0.9356
Run: 6, best_test: 0.9357
Run: 7, best_test: 0.9358
Run: 8, best_test: 0.9361
Run: 9, best_test: 0.9363
avg test f1 score:0.9355 ± 0.0013


python ind-gcn.py --gpu 3 --dataset Reddit --hidden 64 --binarized
Run: 0, best_test: 0.9224
Run: 1, best_test: 0.9256
Run: 2, best_test: 0.9222
Run: 3, best_test: 0.0000
Run: 4, best_test: 0.9232
Run: 5, best_test: 0.9288
Run: 6, best_test: 0.9187
Run: 7, best_test: 0.9146
Run: 8, best_test: 0.9163
Run: 9, best_test: 0.9265
avg test f1 score:0.8298 ± 0.2916
python ind-gcn.py --gpu 3 --dataset Reddit --hidden 128 --binarized
Run: 0, best_test: 0.9318
Run: 1, best_test: 0.9294
Run: 2, best_test: 0.9294
Run: 3, best_test: 0.9303
Run: 4, best_test: 0.9294
Run: 5, best_test: 0.9278
Run: 6, best_test: 0.9271
Run: 7, best_test: 0.9304
Run: 8, best_test: 0.9286
Run: 9, best_test: 0.9274
avg test f1 score:0.9292 ± 0.0015
python ind-gcn.py --gpu 0 --dataset Reddit --hidden 256 --binarized
Run: 0, best_test: 0.9317
Run: 1, best_test: 0.9276
Run: 2, best_test: 0.9270
Run: 3, best_test: 0.9344
Run: 4, best_test: 0.9297
Run: 5, best_test: 0.9279
Run: 6, best_test: 0.9334
Run: 7, best_test: 0.9314
Run: 8, best_test: 0.9322
Run: 9, best_test: 0.9346
avg test f1 score:0.9310 ± 0.0028
python ind-gcn.py --gpu 9 --dataset Reddit --hidden 512 --binarized
Run: 0, best_test: 0.9294
Run: 1, best_test: 0.9314
Run: 2, best_test: 0.9319
Run: 3, best_test: 0.9312
Run: 4, best_test: 0.9282
Run: 5, best_test: 0.9309
Run: 6, best_test: 0.9292
Run: 7, best_test: 0.9327
Run: 8, best_test: 0.9333
Run: 9, best_test: 0.9289
avg test f1 score:0.9307 ± 0.0017
python ind-gcn.py --gpu 4 --dataset Reddit --hidden 1024 --binarized
Run: 0, best_test: 0.9309
Run: 1, best_test: 0.9338
Run: 2, best_test: 0.9329
Run: 3, best_test: 0.9323
Run: 4, best_test: 0.9319
Run: 5, best_test: 0.9337
Run: 6, best_test: 0.9305
Run: 7, best_test: 0.9345
Run: 8, best_test: 0.9291
Run: 9, best_test: 0.9302
avg test f1 score:0.9320 ± 0.0018
```

