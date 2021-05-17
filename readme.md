# Bi-GCN
Official Implementation of CVPR 2021 Paper: [Bi-GCN: Binary Graph Convolutional Network](https://arxiv.org/abs/2010.07565)

Please cite our paper if you use this code in your own work:

```
@article{wang2020bi,
  title={Bi-GCN: Binary Graph Convolutional Network},
  author={Wang, Junfu and Wang, Yunhong and Yang, Zhen and Yang, Liang and Guo, Yuanfang},
  journal={arXiv preprint arXiv:2010.07565},
  year={2020}
}
```

## Requirements
- torch==1.7.0
- torch_geometric==1.7.0
- scikit_learn

## Run

Run the demo of Bi-GCN on Cora dataset by this command.
```
python transductive-bigcn.py --device 0
```

You can specify a dataset, set the layer number, or other hyper-parameters by setting the optional args.

```
python bi-gcn.py --gpu 0 --dataset Cora --layers 4
```
You can run the file `inductive-gs-bignn.py` and `inductive-ns-bignn.py` to get the results of binarized version of other GNNs, like inductive GCN, GraphSAGE, and GraphSAINT.

```bash
python inductive-ns-bignn.py --device 6 --model GraphSAGE --dataset Reddit --binarize
```

The shell script of the reported results in Table 2, 3 can be found in `results.sh`.
