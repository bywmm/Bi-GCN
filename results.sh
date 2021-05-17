eoch "===================Transductive================================="
echo "Cora Bi-GCN"
python transductive-bigcn.py --device 0 --dataset Cora

echo "PubMed Bi-GCN"
python transductive-bigcn.py --device 0 --dataset PubMed


eoch "===================Inductive===================================="
echo "Reddit Bi-indGCN"
# 93.1 ¡À 0.2
python inductive-ns-bignn.py --device 7 --model indGCN --dataset Reddit --binarize

echo "Flickr Bi-indGCN"
# 50.2 ¡À 0.4
python inductive-ns-bignn.py --device 7 --model indGCN --dataset Flickr --batch 1024 --epochs 20 --binarize

echo "Reddit Bi-GraphSAGE"
# 95.3 ¡À 0.1
python inductive-ns-bignn.py --device 5 --model GraphSAGE --dataset Reddit --binarize

echo "Flickr Bi-GraphSAGE"
# 50.2 ¡À 0.4    0.5045 ¡À 0.0055
python inductive-ns-bignn.py --device 5 --model GraphSAGE --dataset Flickr --batch 1024 --hidden 256 --epochs 20 --binarize

echo "Reddit Bi-GraphSAINT"
# 95.7 ¡À 0.1
python inductive-gs-bignn.py --device 9 --dataset Reddit --batch 2000 --walk_length 4 --sample_coverage 50 --epochs 100 --lr 0.003 --hidden 128 --dropout 0.2 --binarize

echo "Flickr Bi-GraphSAINT"
# 50.8 ¡À 0.2
python inductive-gs-bignn.py --device 9 --dataset Flickr --batch 6000 --walk_length 6 --sample_coverage 100 --epochs 50 --lr 0.003 --hidden 256 --dropout 0.4 --binarize
