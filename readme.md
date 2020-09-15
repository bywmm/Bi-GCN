1. 安装依赖
2. 修改bi-gcn的root为你的dataset路径
3. 运行程序

## Transductive
Run the following command to get the results of Bi-GCN on three citation networks.
```
python bi-gcn.py --gpu 0 --dataset Cora
python bi-gcn.py --gpu 0 --dataset CiteSeer
python bi-gcn.py --gpu 0 --dataset PubMed
```

### Model Depth
You can set the `--layers` parameters to get the Bi-GCN with coresponding model depths.s
```
python bi-gcn.py --gpu 0 --dataset Cora --layers 4
```

## Inductive