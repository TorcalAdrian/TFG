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

def load_data_(file_path):
    df = pd.read_csv(file_path, sep=';')
    df = df.drop(columns=['ID'])
    return df

def load_data(file_path):
    classes = {}
    transactions = {}

    with open(file_path, mode='rt', encoding='UTF-8') as file:
        next(file)
        for label, line in enumerate(file):
            parts = line.strip().split(';')
            class_ = int(parts[1])
            events = parts[2].split()
            classes[label] = class_
            transactions[label] = events
            if label==10:
                break

    return classes, transactions



def preprocess_data(df):
    columns_to_use = df.columns[df.columns != 'CLASS']
    df['processed_text'] = df[columns_to_use].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    df['processed_text'] = df['processed_text'].apply(lambda x: x.split())
    return df

def train_word2vec_model(sentences, vector_size, window, epochs):
    start_time = time.time()
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1, workers=4, sg=1, epochs=epochs)
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time

def get_average_vector(words, model_dw):
    vectors = []
    for word in words:
        print(f"word: {word}")
        if word in model_dw.wv:
            vectors.append(model_dw.wv[word])
    if vectors:
        return normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
    else:
        return np.zeros(model_dw.vector_size)

def calculate_normalized_centroids(model, labelled_transactions):
    print("Calculating centroids...")
    dim = model.wv[labelled_transactions[0][0]].shape[0]
    num_labels = len(labelled_transactions)
    centroids = np.zeros((num_labels, dim)) 
    for i, label in enumerate(tqdm(labelled_transactions)):
        word_vectors = np.array([model.wv   [it] for it in labelled_transactions[label] if it in model.wv])

        if word_vectors.size > 0:
            mean_vector = np.mean(word_vectors, axis=0)
            centroids[i] = normalize(mean_vector.reshape(1, -1), norm='l2').reshape(dim,)
    
    return centroids







def run_kmeans_clustering(file_path, vector_size, window, epochs):
    results = []
    classes, events = load_data(file_path)
    real_labels_numeric = [classes[label] for label in sorted(classes.keys())]

    # df= load_data_(file_path)
    # df = preprocess_data(df)
    # model_dw, training_time = train_word2vec_model(events, vector_size=vector_size, window=window, epochs=epochs)
    model_path=f"../../models/{os.path.basename(file_path).replace('.data', '.model')}"
    model_dw = Word2Vec.load(model_path)
    print(f"Model loaded from {model_path}")
    X=calculate_normalized_centroids(model_dw, events)

    print(f"X shape: {X.shape}")
    # if os.path.basename(file_path) == "db36-sep_w2v.data":
    #     n_clusters_settings = [16]
    # elif os.path.basename(file_path) == "db2014-sep_w2v.data":
    #     n_clusters_settings = [16]
    # elif os.path.basename(file_path) == "db201610-sep_w2v.data":
    #     n_clusters_settings = [16]

    n_clusters_settings = [4]
    for n_clusters in n_clusters_settings:
        iteration_results = []
        for i in range(5): 
            print("starting kmeans")
            start_time = datetime.datetime.now()
            # kmeans = KMeans(n_clusters=n_clusters, n_init=1)
            # kmeans.fit(X)
            dim=X.shape[1]
            X = X.astype(np.float32)
            kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=1, verbose=True)
            kmeans.train(X)
            end_time = datetime.datetime.now()
            running_time = (end_time - start_time).total_seconds()
            print("finishing kmeans")
            y_kmeans = kmeans.index.search(X.astype(np.float32), 1)[1].flatten()
            # y_kmeans = kmeans.predict(X)
            print(running_time)
            # Calculate NMI
  
            nmi = normalized_mutual_info_score(real_labels_numeric, y_kmeans)

            iteration_results.append({
                "file": os.path.basename(file_path).replace('.data', ''),
                "n_clusters": n_clusters,
                "vector_size": vector_size,
                "window": window,
                "epochs": epochs,
                "NMI": nmi,
                "running_time": running_time,
                "training_time": 1296,
                "iteration": i + 1  
            })

        nmi_scores = [result["NMI"] for result in iteration_results]
        running_times = [result["running_time"] for result in iteration_results]
        mean_nmi = np.mean(nmi_scores)
        mean_running_time = np.mean(running_times)

        mean_result = iteration_results[0].copy()
        mean_result["NMI"] = mean_nmi
        mean_result["running_time"] = mean_running_time
        mean_result["iteration"] = "mean"

        results.extend(iteration_results)
        results.append(mean_result)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_folder = "../results/results_raw_linux"
    os.makedirs(results_folder, exist_ok=True)
    result_file_path = os.path.join(results_folder, f"results_{os.path.basename(file_path)}.csv")

    if os.path.exists(result_file_path):
        existing_df = pd.read_csv(result_file_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)

    results_df.to_csv(result_file_path, index=False)

    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))