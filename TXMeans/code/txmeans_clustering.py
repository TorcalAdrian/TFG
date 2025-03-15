from algorithms.txmeans import *
from generators.datamanager import *
from validation.validation_measures import *



def run_txmeans_clustering(file_path):
    txmeans = TXmeans()
    class_index = 0
    skipcolumnsindex = set()
    baskets_real_labels, maps = read_uci_data(file_path, class_index=class_index, skipcolumnsindex=skipcolumnsindex)
    baskets_list = list()
    real_labels = list()
    count = 0
    for basket, label in baskets_real_labels:
        baskets_list.append(basket)
        real_labels.append(label)
        count += 1

    baskets_list, map_newitem_item, map_item_newitem = remap_items(baskets_list)
    baskets_list = basket_list_to_bitarray(baskets_list, len(map_newitem_item))

    nbaskets = len(baskets_list)
    nitems = count_items(baskets_list)

    start_time = datetime.datetime.now()

    nsample = sample_size(nbaskets, 0.05, conf_level=0.99, prob=0.5)
    txmeans.fit(baskets_list, nbaskets, nitems, random_sample=nsample)


    end_time = datetime.datetime.now()
    running_time = end_time - start_time

    res = txmeans.clustering
    pred_labels = [0] * len(real_labels)
    baskets_clusters = list()
    for cluster, label in zip(res, range(0, len(res))):
        cluster_list = basket_bitarray_to_list(cluster['cluster']).values()
        for bid in cluster['cluster']:
            pred_labels[bid] = label
            baskets_clusters.append(cluster_list)
    normalized_mutual_info_score(real_labels, pred_labels)


