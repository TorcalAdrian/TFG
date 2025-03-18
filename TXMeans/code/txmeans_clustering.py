from algorithms.txmeans import *
from validation.validation_measures import *
import pandas as pd
import numpy as np
from scipy.stats import mode
import sys




def read_uci_data(filename, class_index=0, delimiter=',', missing_symbol='?', header=True, skipcolumnsindex=set(),events_index=2):

    df = pd.read_csv(filename, delimiter=delimiter, skipinitialspace=True)
    index_mode = {}

    for k, index in zip(df.columns, range(len(df.columns))):
        df[k] = df[k].replace(missing_symbol, np.nan)
        mode_value = mode(df[k].dropna())[0][0]
        df[k] = df[k].fillna(mode_value)
        index_mode[index] = mode_value

    baskets = []
    
    map_item_newitem = {}
    map_newitem_item = {}
    map_class_newclass = {}
    map_newclass_class = {}
    
    # Abrir el archivo para procesarlo línea a línea
    with open(filename, 'r') as data:
        if header:
            data.readline()  # Saltar la cabecera
        for row in data:
            categories = row.rstrip().split(delimiter)
            basket = []
            basket_class = None
            
            # Recorrer cada columna de la fila
            for index in range(len(categories)):
                # Saltar columnas que se deben omitir
                if index in skipcolumnsindex:
                    continue
                
                # Procesar la columna de la clase
                if index == class_index:
                    cclass = categories[index]
                    if cclass not in map_class_newclass:
                        newclass = len(map_class_newclass)
                        map_class_newclass[cclass] = newclass
                        map_newclass_class[newclass] = cclass
                    basket_class = map_class_newclass[cclass]
                    continue
                

                if index == events_index:
                    if categories[index] == missing_symbol:
                        categories[index] = index_mode[index]

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

                    if categories[index] == missing_symbol:
                        categories[index] = index_mode[index]
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


def main():
    
    sys.path.insert(0, r'/home/adrian/Escritorio/TFG/TXMeans/code/')
    path = '../../dataset_pp/'
    dataset_name = 'P0.O0_tx.data'

    txmeans = TXmeans()
    
    filename = path + dataset_name
    class_index = 1
    skipcolumnsindex = set({0})
    
    baskets_real_labels, maps = read_uci_data(filename, class_index=class_index,delimiter=";", skipcolumnsindex=skipcolumnsindex)

    print( dataset_name, len(baskets_real_labels))

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

    print('delta_k', delta_k(real_labels, pred_labels))
    print('normalized_mutual_info_score', normalized_mutual_info_score(real_labels, pred_labels))
    print('purity', purity(real_labels, pred_labels))
    print('running_time', running_time)
    

if __name__ == "__main__":
    main()
