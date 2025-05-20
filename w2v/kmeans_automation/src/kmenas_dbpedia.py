import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
import faiss

import datetime
from tabulate import tabulate
import csv
import os
import time
from tqdm import tqdm
from joblib import Parallel, delayed


def load_data(file_path,class_path):
    classes = []
    transactions = []
    
    with open(class_path, mode='rt', encoding='UTF-8') as file:
        next(file)
        for line in file:
            parts = line.strip().split(';')
            class_ = parts[1]
            classes.append(class_)
    
    transactions = np.load(file_path)


    return transactions, classes







def run_kmeans_clustering(file_path, class_path,vector_size, window, epochs, training_time,embeding_time):
    events, classes = load_data(file_path,class_path)

    iterations = 20
    
    basename = os.path.basename(class_path)

    output_csv = '../../../final_results/w2v_txmeans_cluster.csv'
    clusters_dict = {   
            "adult_tx": 2,
            "chessBig_tx": 14,
            "connect_tx": 3,
            "db36-sep_w2v.data": 10,
            "db2014-sep_w2v.data": 13,
            "db201610-sep_w2v.data": 10,
            "letrecog_tx": 10,
            "P0.O0_tx": 6,
            "P0.O30_tx": 6,
            "P20.O0_tx": 7,
            "P20.O30_tx": 7,
            "P50.O0_tx": 6,
            "P50.O30_tx": 6,
            "pendigits_tx": 24,
            "T500k.D20k.L50.P60.O40.C8": 1
        }

    file_exists = os.path.isfile(output_csv)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['filename', 'n_clusters','NMI', 'running_time', 'training_time','embeding_time','iteration'])

        n_clusters = clusters_dict[basename]
        # for n_clusters in n_clusters_settings:
        nmi_list = []
        nclusters_list = []
        training_time_list = []
        embeding_time_list = []
        running_time_list = []

        for i in range(iterations): 
            print(f"Ejecutando iteración {i + 1} de {iterations}")
            start_time = datetime.datetime.now()
            dim=events.shape [1]
            kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=1, verbose=False)
            kmeans.train(events)
            y_kmeans = kmeans.index.search(events.astype(np.float32), 1)[1].flatten()
            end_time = datetime.datetime.now()
            running_time = (end_time - start_time).total_seconds()


            nmi = normalized_mutual_info_score(classes, y_kmeans)

            nmi_list.append(nmi)
            nclusters_list.append(n_clusters)
            embeding_time_list.append(embeding_time)
            running_time_list.append(running_time)
            training_time_list.append(training_time)
            writer.writerow([basename, n_clusters,nmi, running_time, training_time, embeding_time,i+1])
        avg_nmi = sum(nmi_list) / iterations
        avg_nclusters = "Nan"
        training_time_deltak = sum(training_time_list) / iterations
        avg_running_time = sum(running_time_list) / iterations
        embeding_time_time = sum(embeding_time_list) / iterations
        writer.writerow([basename, avg_nclusters,avg_nmi, avg_running_time, training_time_deltak, embeding_time_time, 'avg'])

            




def main():
    dataset_folder = "../../../datasets/w2v_transactions"
    class_folder = "../../../datasets/dataset_db_w2v"


    
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.npy') :
            clas_name = f"{filename.split('_')[0]}_w2v.data"
            class_path = os.path.join(class_folder, clas_name)
            print(class_path)
            file_path = os.path.join(dataset_folder, filename)
            name=filename.replace('.npy','').split('_')
            vector_size=int(name[2])
            window=int(name[3])
            epochs=int(name[4])
            runing_time=float(name[5])
            embeading_time=float(name[6])
            print(os.path.basename(class_path))
            run_kmeans_clustering(file_path,class_path, vector_size, window, epochs, runing_time,embeading_time)
              
            

if __name__ == "__main__":
    main()
