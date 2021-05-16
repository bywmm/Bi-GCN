1. Change the variable `root` to your dataset root.
2. Create an Anaconda environment use `requirements.txt`.
```
conda env create -f geometric.yaml
```

## Transductive
Run the following command to get the results of Bi-GCN on three citation networks.
```
python bi-gcn.py --gpu 0 --dataset Cora
python bi-gcn.py --gpu 0 --dataset CiteSeer
python bi-gcn.py --gpu 0 --dataset PubMed
```

### Model Depth
You can set the `--layers` parameters to get the Bi-GCN with coresponding model depths.
```
python bi-gcn.py --gpu 0 --dataset Cora --layers 4
```

## Inductive
Run the following command to get the results of our binarized version of inductive GCN, GraphSAGE, and GraphSAINT on the Reddit and Flickr datasets.

```
echo "Reddit Bi-IndGCN"
python ind-gcn.py --gpu 7 --dataset Reddit --binarized

echo "Flickr Bi-IndGCN"
python ind-gcn.py --gpu 8 --dataset Flickr --epochs 20 --binarized

echo "Reddit Bi-GraphSAGE"
python graphsage.py --gpu 7 --dataset Reddit --binarized

echo "Flickr Bi-GraphSAGE"
python graphsage.py --gpu 4 --epochs 20 --dataset Flickr --binarized

echo "Reddit Bi-GraphSAINT"
python graphsaint.py --gpu 8 --dataset Reddit --batch 2000 --walk_length 4 --sample_coverage 50 --epochs 100 --lr 0.003 --hidden 128 --dropout 0.2 --binarized

echo "Flickr Bi-GraphSAINT"
python graphsaint.py --gpu 7 --dataset Flickr --batch 6000 --walk_length 6 --sample_coverage 100 --epochs 50 --lr 0.003 --hidden 256 --dropout 0.4 --binarized
```