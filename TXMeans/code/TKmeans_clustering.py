import sys
sys.path.insert(0, r'/home/adrian/Escritorio/TFG/TXMeans/code/')

from algorithms.tkmeans import *
from generators.datamanager import *
from validation.validation_measures import *

from algorithms.tkmeans import *
import csv
import os
from tqdm import tqdm
import datetime
import signal




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

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException


def run_txmeans_clustering(filename):
    iterations = 20
    nmi_list = []
    nclusters_list = []
    deltak_list = []
    running_time_list = []
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    output_csv = '../../final_results/tkmeans_all_datasets.csv'

    file_exists = os.path.isfile(output_csv)
    clusters_list = [4, 8, 16]
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['filename', 'nmi', 'deltak','nclusters', 'running_time', 'iteration'])
        for nclusters in clusters_list:
            nmi_list = []
            deltak_list = []
            running_time_list = []
            for i in range(iterations):
                print(f"Ejecutando iteración {i + 1} de {iterations}")
                tkmeans = TKMeans()
                nitems = count_items(filename)

                class_index = 1
                skipcolumnsindex = set({0})
                    
                baskets_real_labels, maps = read_uci_data(filename, class_index=class_index,delimiter=";", skipcolumnsindex=skipcolumnsindex)

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
                start_time = datetime.datetime.now()


                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30 * 60)  # Set timeout to 30 minutes

                try:
                    start_time = datetime.datetime.now()
                    tkmeans.fit(baskets_list, nbaskets, nitems, nclusters)
                    end_time = datetime.datetime.now()
                    running_time = end_time - start_time

                    res = tkmeans.clustering
                    #iter_count = bicartd.iter_count
                    pred_labels = [0] * len(real_labels)
                    baskets_clusters = list()
                    for cluster, label in zip(res, range(0, len(res))):
                        cluster_list = basket_bitarray_to_list(cluster['cluster']).values()
                        for bid in cluster['cluster']:
                            pred_labels[bid] = label
                            baskets_clusters.append(cluster_list)


                    nmi = normalized_mutual_info_score(real_labels, pred_labels)
                    deltak = delta_k(real_labels, pred_labels)
                    # purity_score = purity(real_labels, pred_labels)
                    running_time_seconds = running_time.total_seconds()
                    running_time_seconds=running_time_seconds
                    nmi_list.append(nmi)
                    deltak_list.append(deltak)
                    running_time_list.append(running_time_seconds)
                    writer.writerow([basename, nmi, deltak, len(res), running_time_seconds,i+1])
                except TimeoutException:
                    writer.writerow([basename, "skip", "skip", "skip", "+30mins",i+1])
                    print("Timeout reached. Skipping to the next iteration.")
                finally:
                    signal.alarm(0)  # Disable the alarm

                

            avg_nmi = sum(nmi_list) / iterations
            avg_deltak = sum(deltak_list) / iterations
            avg_running_time = sum(running_time_list) / iterations
            writer.writerow([basename, avg_nmi, avg_deltak, nclusters, avg_running_time, 'avg'])
        




        
        

