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



def load_data(file_path):
    class_ = []
    transactions = []
    with open(file_path, mode='rt', encoding='UTF-8') as file:
        next(file)
        for line in file:
            parts = line.strip().split(';')
            classes_= parts[1].split()
            events = parts[2].split()
            class_.append(classes_)
            transactions.append(events)

    return class_,transactions



def train_word2vec_model(sentences, vector_size, window, epochs):
    start_time = time.time()
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1, workers=4, sg=1, epochs=epochs)
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time



def calculate_normalized_centroids(model, labelled_transactions):
    start_time = time.time()
    dim = model.wv[labelled_transactions[0][0]].shape[0]
    num_labels = len(labelled_transactions)
    centroids = np.zeros((num_labels, dim))
    for i,transaction in enumerate(labelled_transactions):
        word_vectors = np.array([model.wv   [it] for it in transaction if it in model.wv])
        if word_vectors.size > 0:
            mean_vector = np.mean(word_vectors, axis=0)
            centroids[i] = normalize(mean_vector.reshape(1, -1), norm='l2').reshape(dim,)

    end_time = time.time()
    training_time = end_time - start_time
    return centroids, training_time







def run_kmeans_clustering(file_path, vector_size, window, epochs):

    classes, events = load_data(file_path)
    iterations = 20
    
    basename = os.path.basename(file_path)
    basename = os.path.splitext(basename)[0]
    output_csv = '../../../final_results/w2v_txmeans_cluster.csv'


    file_exists = os.path.isfile(output_csv)
    
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['filename', 'n_clusters','NMI', 'running_time', 'training_time','embeding_time','iteration'])
        clusters_dict = {   
            "adult_tx": 4,
            "chessBig_tx": 14,
            "connect_tx": 3,
            "db36-sep_w2v": 10,
            "db2014-sep_w2v": 13,
            "db201610-sep_w2v": 10,
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

        nmi_list = []
        nclusters_list = []
        training_time_list = []
        embeding_time_list = []
        running_time_list = []
        n_clusters = clusters_dict[basename]
        print(n_clusters)
        model_dw, training_time = train_word2vec_model(events, vector_size=vector_size, window=window, epochs=epochs)
        X,embeding_time=calculate_normalized_centroids(model_dw, events)
        for i in range(iterations): 
            print(f"Ejecutando iteración {i + 1} de {iterations}")
            # model_dw, training_time = train_word2vec_model(events, vector_size=vector_size, window=window, epochs=epochs)
            # X,embeding_time=calculate_normalized_centroids(model_dw, events)
            start_time = datetime.datetime.now()
            # kmeans = KMeans(n_clusters=n_clusters, n_init=1)
            # kmeans.fit(X)
            dim=X.shape[1]
            X = X.astype(np.float32)
            kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=1, verbose=False)
            kmeans.train(X)
            y_kmeans = kmeans.index.search(X.astype(np.float32), 1)[1].flatten()
            end_time = datetime.datetime.now()
            running_time = (end_time - start_time).total_seconds()


            # y_kmeans = kmeans.predict(X)

            nmi = normalized_mutual_info_score(np.ravel(classes), y_kmeans)

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
    dataset_folder = "../../../datasets/final_datasets/"
    


    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data') and filename!= "db36-sep_w2v.data" and filename== "db2014-sep_w2v.data" and filename== "db201610-sep_w2v.data" :
            file_path = os.path.join(dataset_folder, filename)
            run_kmeans_clustering(file_path, 200, 5, 5)
            # settings_combinations = product(window_settings, vector_size_settings, epochs_settings)

            # for window, vector_size, epochs in settings_combinations:
            #     run_kmeans_clustering(file_path, vector_size, window, epochs)
              
            

if __name__ == "__main__":
    main()
            