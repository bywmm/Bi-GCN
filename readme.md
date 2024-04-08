# Bi-GCN
Official Implementation of CVPR 2021 Paper: [Bi-GCN: Binary Graph Convolutional Network](https://arxiv.org/abs/2010.07565, and TPAMI 2024 Paper: [Binary Graph Convolutional Network With Capacity Exploration](https://ieeexplore.ieee.org/abstract/document/10356827).

Please cite our paper if you use this code in your own work:

```
@INPROCEEDINGS{wang2021,
  author={Wang, Junfu and Wang, Yunhong and Yang, Zhen and Yang, Liang and Guo, Yuanfang},
  booktitle={2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Bi-GCN: Binary Graph Convolutional Network}, 
  year={2021},
  pages={1561-1570}
```

```
@ARTICLE{wang2023,
  author={Wang, Junfu and Guo, Yuanfang and Yang, Liang and Wang, Yunhong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Binary Graph Convolutional Network With Capacity Exploration}, 
  year={2024},
  volume={46},
  number={5},
  pages={3031-3046}
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
