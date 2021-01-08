import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')  # Cora/CiteSeer/PubMed
parser.add_argument('--hidden', type=int, default=64)
args = parser.parse_args()

hidden = args.hidden
dataset = args.dataset

assert dataset in {'Cora', 'CiteSeer', 'PubMed', 'Reddit', 'Flickr'}, 'Please check the dataset name.'

if dataset == 'Cora':
    num_node = 2708
    num_edge = 5429
    num_class = 7
    num_feature = 1433
elif dataset == 'CiteSeer':
    num_node = 3327
    num_edge = 4732
    num_class = 6
    num_feature = 3703
elif dataset == 'PubMed':
    num_node = 19711
    num_edge = 44338
    num_class = 3
    num_feature = 500
elif dataset == 'Reddit':
    num_node = 232965
    num_edge = 11606919
    num_class = 41
    num_feature = 602
elif dataset =='Flickr':
    num_node = 89250
    num_edge = 899756
    num_class = 7
    num_feature = 500

data_size = num_node * num_feature * 4.0 / 1024 /1024
bi_data_size = (num_node * num_feature / 8 + num_node * 4) /1024/1024
print("Data Size: {:.2f}M".format(data_size))
print("Binarized Data Size: {:.2f}M".format(bi_data_size))
print("Data size ratio : {:.2f}".format(data_size / bi_data_size))

if dataset in {'Cora', 'CiteSeer', 'PubMed'}:
    print("Model Size")
    GCN_ms = (num_feature * hidden + hidden * num_class) * 4 / 1024
    print("GCN: {:.2f}K".format(GCN_ms))
    biGCN_ms = ((num_feature * hidden + hidden * num_class) / 8 + hidden * 4 + num_class * 4) / 1024
    print("Bi-GCN: {:.2f}K".format(biGCN_ms))
    print("Model size ratio : {:.2f}".format(GCN_ms / biGCN_ms))
    # Cal cycle operations
    GCN_co = num_node * num_feature * hidden + num_edge * hidden + \
    num_node * hidden * num_class + num_edge * num_class
    print("GCN o.p.: ", GCN_co)
    GAT_co = GCN_co + num_edge * (2 * hidden + 2 * num_class + 8 + 1)
    print("GAT o.p.: ", GAT_co)
    SGC_co = num_node * num_feature * num_class + num_edge * num_class * 2
    print("SGC o.p.: ", SGC_co)
    BiGCN_co = num_node * num_feature * hidden / 64 + 2 * num_node * hidden + num_edge * hidden + \
    num_node * hidden * num_class / 64 + 2 * num_node * num_class + num_edge * num_class
    print("Bi-GCN o.p.: ", BiGCN_co)
    print("Cycle operation ratio : {:.2f}".format(GCN_co / BiGCN_co))

else:
    hidden = 256
    print("=========================")
    print("Model Size")
    indGCN_ms = (num_feature * hidden + hidden * num_class) * 4 / 1024
    print("indGCN: {:.2f}K".format(indGCN_ms))
    bi_indGCN_ms = ((num_feature * hidden + hidden * num_class) / 8 + (hidden+num_class)*4) / 1024
    print("bi-indGCN: {:.2f}K".format(bi_indGCN_ms))
    print("bi-indGCN ratio : {:.2f}".format(indGCN_ms / bi_indGCN_ms))

    sage_ms = (num_feature * hidden + hidden * num_class) * 2 * 4 / 1024
    print("sage: {:.2f}K".format(sage_ms))
    bi_sage_ms = ((num_feature * hidden + hidden * num_class) / 8 + (hidden+num_class)*4) * 2 / 1024
    print("bi-sage: {:.2f}K".format(bi_sage_ms))
    print("bi-sage ratio : {:.2f}".format(sage_ms / bi_sage_ms))

    saint_ms = ((num_feature * hidden + hidden * hidden) * 2 + 2 * hidden * num_class) * 4 / 1024
    print("saint: {:.2f}K".format(saint_ms))
    bi_saint_ms = ((num_feature * hidden + hidden * hidden) / 8 + (hidden+hidden)*4) * 2 / 1024 +\
        (2 * hidden * num_class) * 4 / 1024
    print("bi-saint: {:.2f}K".format(bi_saint_ms))
    print("bi-saint ratio : {:.2f}".format(saint_ms / bi_saint_ms))

    print("=========================")
    print('Cycle Operations')
    indGCN_co = num_node * num_feature * hidden + num_edge * hidden + \
             num_node * hidden * num_class + num_edge * num_class
    print("indGCN: ", indGCN_co)
    bi_indGCN_co = num_node * num_feature * hidden / 64 + 2 * num_node * hidden + num_edge * hidden + \
                 num_node * hidden * num_class / 64 + 2 * num_node * num_class + num_edge * num_class
    print("bi-indGCN: ", bi_indGCN_co)
    print("bi-indGCN ratio : {:.2f}".format(indGCN_co / bi_indGCN_co))

    sage_co = num_node * num_feature * hidden * 2 + num_edge * hidden + \
             num_node * hidden * num_class * 2 + num_edge * num_class
    print("sage: ", sage_co)
    bi_sage_co = num_node * num_feature * hidden / 64 * 2 + 2 * num_node * hidden * 2 + num_edge * hidden + \
                 num_node * hidden * num_class / 64 * 2 + 2 * num_node * num_class * 2+ num_edge * num_class
    print("bi-sage: ", bi_sage_co)
    print("bi-sage ratio : {:.2f}".format(sage_co / bi_sage_co))

    saint_co = num_node * num_feature * hidden * 2 + num_edge * hidden + \
               num_node * hidden * hidden * 2 + num_edge * hidden + \
               num_node * hidden * 2 * num_class
    print("saint: ", saint_co)
    bi_saint_co = num_node * num_feature * hidden / 64 * 2 + 2 * num_node * hidden * 2 + num_edge * hidden + \
                  num_node * hidden * hidden / 64 * 2 + 2 * num_node * hidden * 2 + num_edge * num_class + \
                  num_node * hidden * 2 * num_class
    print("bi-saint: ", bi_saint_co)
    print("bi-saint ratio : {:.2f}".format(saint_co / bi_saint_co))
