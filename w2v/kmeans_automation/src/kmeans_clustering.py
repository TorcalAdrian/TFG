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
    output_csv = '../../../final_results/w2v_all_datasets.csv'


    file_exists = os.path.isfile(output_csv)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['filename', 'n_clusters','NMI', 'running_time', 'training_time','embeding_time','iteration'])

        n_clusters_settings = [4,8,16]
        for n_clusters in n_clusters_settings:
            nmi_list = []
            nclusters_list = []
            training_time_list = []
            embeding_time_list = []
            running_time_list = []
            

            for i in range(iterations): 
                print(f"Ejecutando iteración {i + 1} de {iterations}")
                model_dw, training_time = train_word2vec_model(events, vector_size=vector_size, window=window, epochs=epochs)
                X,embeding_time=calculate_normalized_centroids(model_dw, events)
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

                del model_dw
                del X
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

                

            