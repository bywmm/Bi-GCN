#!/bin/sh

gpuid='8'

echo "IndGCN"
echo "===="

echo "Reddit IndGCN"
python ind-gcn.py --gpu 8 --dataset Reddit
echo "Reddit Bi-IndGCN"
python ind-gcn.py --gpu 7 --dataset Reddit --binarized

echo "Flickr IndGCN"
python ind-gcn.py --gpu 8 --dataset Flickr --epochs 20
echo "Flickr Bi-IndGCN"
python ind-gcn.py --gpu 8 --dataset Flickr --epochs 20 --binarized

echo "GraphSAGE"
echo "===="

echo "Reddit GraphSAGE"
python graphsage.py --gpu 8 --dataset Reddit
echo "Reddit Bi-GraphSAGE"
python graphsage.py --gpu 7 --dataset Reddit --binarized

echo "Flickr GraphSAGE"
python graphsage.py --gpu 6 --epochs 20 --dataset Flickr
echo "Flickr Bi-GraphSAGE"
python graphsage.py --gpu 4 --epochs 20 --dataset Flickr --binarized

echo "GraphSAINT"
echo "===="

echo "Reddit GraphSAINT"
python graphsaint.py --gpu 3 --dataset 'Reddit' --batch 2000 --walk_length 4 --sample_coverage 50 --epochs 100 --lr 0.01 --hidden 128 --dropout 0.1
echo "Reddit Bi-GraphSAINT"
python graphsaint.py --gpu 4 --dataset 'Reddit' --batch 2000 --walk_length 4 --sample_coverage 50 --epochs 100 --lr 0.01 --hidden 128 --dropout 0.1 --binarized

echo "Flickr GraphSAINT"
python graphsaint.py --gpu 8 --dataset 'Flickr' --batch 6000 --walk_length 6 --sample_coverage 100 --epochs 50 --lr 0.001 --hidden 256 --dropout 0.2
echo "Flickr Bi-GraphSAINT"
python graphsaint.py --gpu 7 --dataset 'Flickr' --batch 6000 --walk_length 6 --sample_coverage 100 --epochs 50 --lr 0.001 --hidden 256 --dropout 0.2 --binarized

echo