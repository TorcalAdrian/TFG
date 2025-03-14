import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
from tabulate import tabulate
import os
import time

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    df = df.drop(columns=['ID'])
    return df

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



def compute_tfidf_weights(texts):

    texts_strings = [" ".join(words) for words in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts_strings)
    tfidf_vocab = vectorizer.get_feature_names_out()
    tfidf_scores = dict(zip(tfidf_vocab, np.array(tfidf_matrix.mean(axis=0)).flatten()))
    return tfidf_scores

def get_weighted_vector(words, model, tfidf_scores):

    vectors = []
    weights = []

    for word in words:
        if word in model.wv and word in tfidf_scores:
            vectors.append(model.wv[word] * tfidf_scores[word])  # Ponderación
            weights.append(tfidf_scores[word])

    if vectors:
        weighted_mean = np.average(vectors, axis=0, weights=weights)  # Media ponderada
        return normalize(weighted_mean.reshape(1, -1))[0]  # Normalizar
    else:
        return np.zeros(model.vector_size)

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
    n_clusters_settings = [2, 4, 8,10,16,20,50,100]

    model_dw, training_time = train_word2vec_model(df['processed_text'], vector_size=vector_size, window=window, epochs=epochs)

    tfidf_scores = compute_tfidf_weights(df['processed_text'])

    X = np.array([get_weighted_vector(text, model_dw, tfidf_scores) for text in df['processed_text']])


    for n_clusters in n_clusters_settings:
        iteration_results = []
        for i in range(10): 
            start_time = datetime.datetime.now()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(X)
            end_time = datetime.datetime.now()
            running_time = (end_time - start_time).total_seconds()

            y_kmeans = kmeans.predict(X)
            real_labels_numeric = df['CLASS']
            nmi = normalized_mutual_info_score(real_labels_numeric, y_kmeans)

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
        median_nmi = np.median(nmi_scores)
        median_index = np.argsort(np.abs(np.array(nmi_scores) - median_nmi))[0]
        median_result = iteration_results[median_index].copy()
        median_result["iteration"] = "median"

        # Add all iteration results and the median result to the final results
        results.extend(iteration_results)
        results.append(median_result)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_folder = "../results/results_raw_tf"
    os.makedirs(results_folder, exist_ok=True)
    result_file_path = os.path.join(results_folder, f"results_{os.path.basename(file_path)}.csv")

    if os.path.exists(result_file_path):
        existing_df = pd.read_csv(result_file_path)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)

    results_df.to_csv(result_file_path, index=False)

    # print(tabulate(results, headers="keys", tablefmt="fancy_grid"))