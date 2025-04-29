import sys
sys.path.insert(0, r'/home/adrian/Escritorio/TFG/TXMeans/code/')

from algorithms.txmeans import *
from generators.datamanager import *
from validation.validation_measures import *
# from generators.datagenerator import *
from algorithms.tkmeans import *
import csv
import os
from scipy.sparse import lil_matrix
from tqdm import tqdm





def read_uci_data(filename, class_index=0, delimiter=';', missing_symbol='?', header=True, skipcolumnsindex=set(),events_index=2):


    baskets = []

    map_item_newitem = {}
    map_newitem_item = {}
    map_class_newclass = {}
    map_newclass_class = {}
    
    
    with open(filename, 'r') as data:
        if header:
            data.readline()  
        for row in data:
            categories = row.rstrip().split(delimiter)
            basket = []
            basket_class = None
            

            for index in range(len(categories)):

                if index in skipcolumnsindex:
                    continue

                if index == class_index:
                    
                    cclass = categories[index]
                    if cclass not in map_class_newclass:
                        newclass = len(map_class_newclass)
                        map_class_newclass[cclass] = newclass
                        map_newclass_class[newclass] = cclass
                    basket_class = map_class_newclass[cclass]
                    continue
                

                if index == events_index:
                    events = categories[index].split()

                    for event in events:
                        item = (index, event)
                        if item not in map_item_newitem:
                            newitem = len(map_item_newitem)
                            map_item_newitem[item] = newitem
                            map_newitem_item[newitem] = item
                        newitem = map_item_newitem[item]
                        basket.append(newitem)
                else:
                    print(index)
                    item = (index, categories[index])
                    if item not in map_item_newitem:
                        newitem = len(map_item_newitem)
                        map_item_newitem[item] = newitem
                        map_newitem_item[newitem] = item
                    newitem = map_item_newitem[item]
                    basket.append(newitem)
            if len(basket) > 0:
                baskets.append((basket, basket_class))
    
    maps = {
        'map_item_newitem': map_item_newitem,
        'map_newitem_item': map_newitem_item,
        'map_class_newclass': map_class_newclass,
        'map_newclass_class': map_newclass_class,
    }
    
    return baskets, maps

def count_items(PATH_DATASET_TX):
    f = open(PATH_DATASET_TX  , 'r')
    f.readline()
    all_items = set()
    for line in f:
        trans = line.split(';')[-1]
        items = trans.split()
        all_items.update(items)
    f.close()
    total_items =  len(all_items)
    del all_items
    return total_items



def run_txmeans_clustering(filename):

    txmeans = TXmeans()
    nitems = count_items(filename)

    class_index = 1
    skipcolumnsindex = set({0})
        
    baskets_real_labels, maps = read_uci_data(filename, class_index=class_index,delimiter=";", skipcolumnsindex=skipcolumnsindex)
    del maps
    print("Datos cargados")

    baskets_list = list()
    real_labels = list()
    count = 0
    for basket, label in baskets_real_labels:
        baskets_list.append(basket)
        real_labels.append(label)
        count += 1
    print("Total de transacciones: ", count)
    baskets_list, map_newitem_item, map_item_newitem = remap_items(baskets_list)

    baskets_list = {i: basket for i, basket in enumerate(baskets_list)}
    del map_item_newitem
    print("Total de items: ", nitems)
    # baskets_list = basket_list_to_bitarray(baskets_list, len(map_newitem_item))# se jode la memoria aqui
    nbaskets = len(baskets_list)
    print(nbaskets)
    print(nitems)
    print("comenzando a clusterizar")
    start_time = datetime.datetime.now()
    nsample = sample_size(nbaskets, 0.05, conf_level=0.99, prob=0.5)
    txmeans.fit(baskets_list, nbaskets, nitems, random_sample=nsample)
    print("fin clusterizar")

    res = txmeans.clustering
    pred_labels = [0] * len(real_labels)
    baskets_clusters = list()
    print(len(res))
    for cluster, label in tqdm(zip(res, range(0, len(res)))):
        cluster_list = basket_bitarray_to_list(cluster['cluster']).values()
        for bid in cluster['cluster']:
            pred_labels[bid] = label
            baskets_clusters.append(cluster_list)

    end_time = datetime.datetime.now()
    running_time = end_time - start_time
    print("fin de clusterizar")
    nmi = normalized_mutual_info_score(real_labels, pred_labels)
    deltak = delta_k(real_labels, pred_labels)
    # purity_score = purity(real_labels, pred_labels)
    running_time_seconds = running_time.total_seconds()

    output_csv = 'resultadosTxmeans.csv'

    file_exists = os.path.isfile(output_csv)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['filename', 'nmi', 'deltak', 'running_time'])

        writer.writerow([filename, nmi, deltak,len(res), running_time_seconds])

    print(f"Datos guardados en {output_csv}")

    
    

