import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
import datetime
from tabulate import tabulate
import os

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df = df.drop(columns=['ID'])
    return df

def preprocess_data(df):
    columns_to_use = df.columns[df.columns != 'CLASS']
    df['processed_text'] = df[columns_to_use].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    df['processed_text'] = df['processed_text'].apply(lambda x: x.split())
    return df

def train_word2vec_model(sentences, vector_size=200, window=10, epochs=10):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1, workers=4, sg=1, epochs=epochs)
    return model

def get_average_vector(words, model_dw):
    vectors = []
    for word in words:
        if word in model_dw.wv:
            vectors.append(model_dw.wv[word])
    if vectors:
        return normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
    else:
        return np.zeros(model_dw.vector_size)

def run_kmeans_clustering(file_path, vector_size, window, epochs):
    results = []
    df = load_data(file_path)
    df = preprocess_data(df)
    n_clusters_settings = [2, 4, 8, 10, 16, 20, 50, 100]

    model_dw = train_word2vec_model(df['processed_text'], vector_size=vector_size, window=window, epochs=epochs)
    X = np.array([get_average_vector(text, model_dw) for text in df['processed_text']])

    for n_clusters in n_clusters_settings:
        for i in range(3): 
            start_time = datetime.datetime.now()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X)
            end_time = datetime.datetime.now()
            running_time = (end_time - start_time).total_seconds()

            y_kmeans = kmeans.predict(X)
            real_labels_numeric = df['CLASS']
            nmi = normalized_mutual_info_score(real_labels_numeric, y_kmeans)

            results.append({
                "file": os.path.basename(file_path).replace('.data', ''),
                "n_clusters": n_clusters,
                "vector_size": vector_size,
                "window": window,
                "epochs": epochs,
                "NMI": nmi,
                "running_time": running_time,
                "iteration": i + 1  
            })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_folder = "../results_raw"
    os.makedirs(results_folder, exist_ok=True)
    result_file_path = os.path.join(results_folder, f"results_{os.path.basename(file_path)}.csv")

    if os.path.exists(result_file_path):
        existing_df = pd.read_csv(result_file_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)

    results_df.to_csv(result_file_path, index=False)

    # print(tabulate(results, headers="keys", tablefmt="fancy_grid"))