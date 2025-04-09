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







def run_kmeans_clustering(file_path, class_path,vector_size, window, epochs, training_time):
    results = []
    events, classes = load_data(file_path,class_path)

    

    
    if os.path.basename(class_path) == "db36-sep_w2v.data":
        n_clusters_settings = [505]
    elif os.path.basename(class_path) == "db2014-sep_w2v.data":
        n_clusters_settings = [910]
    elif os.path.basename(class_path) == "db201610-sep_w2v.data":
        n_clusters_settings = [2969]

    # n_clusters_settings = [505,910,2969]
    # n_clusters_settings = [4,16, 32, 64, 128, 256, 512]
    for n_clusters in n_clusters_settings:
        iteration_results = []
        for i in range(5): 
            print("starting kmeans")
            start_time = datetime.datetime.now()
            # kmeans = KMeans(n_clusters=n_clusters, n_init=1)
            # kmeans.fit(X)
            dim=events.shape [1]
            kmeans = faiss.Kmeans(d=dim, k=n_clusters, niter=1, verbose=True)
            kmeans.train(events)
            y_kmeans = kmeans.index.search(events.astype(np.float32), 1)[1].flatten()
            end_time = datetime.datetime.now()
            running_time = (end_time - start_time).total_seconds()
            print("finishing kmeans")
            print(running_time)
            # Calculate NMI
  
            nmi = normalized_mutual_info_score(classes, y_kmeans)

            iteration_results.append({
                "file": os.path.basename(file_path).replace('.data', ''),
                "n_clusters": n_clusters,
                "vector_size": vector_size,
                "window": window,
                "epochs": epochs,
                "NMI": nmi,
                "running_time": running_time,
                "training_time": training_time,
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
    file_path=class_path.split('_')[0]
    result_file_path = os.path.join(results_folder, f"results_{os.path.basename(file_path).replace('.data', '')}.csv")

    if os.path.exists(result_file_path):
        existing_df = pd.read_csv(result_file_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)

    results_df.to_csv(result_file_path, index=False)

    print(tabulate(results, headers="keys", tablefmt="fancy_grid"))





def main():
    dataset_folder = "../../../datasets/w2v_transactions"
    class_folder = "../../../datasets/dataset_db_w2v"


    
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.npy'):
            clas_name = f"{filename.split('_')[0]}_w2v.data"
            class_path = os.path.join(class_folder, clas_name)
            print(class_path)
            file_path = os.path.join(dataset_folder, filename)
            name=filename.replace('.npy','').split('_')
            vector_size=int(name[2])
            window=int(name[3])
            epochs=int(name[4])
            runing_time=float(name[5])
            if epochs==5:
                run_kmeans_clustering(file_path,class_path, vector_size, window, epochs, runing_time)
              
            

if __name__ == "__main__":
    main()
