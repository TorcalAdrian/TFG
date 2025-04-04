import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize
import datetime
from tabulate import tabulate
import os
import time
from itertools import product
from tqdm import tqdm

def load_data(file_path):
    transactions = []
    classes = []

    with open(file_path, mode='rt', encoding='UTF-8') as file:
        next(file)
        for line in file:
            parts = line.strip().split(';')
            class_ = parts[0]
            events = parts[1].split()
            transactions.append(events)
            classes.append(class_)

    return transactions, classes

def train_word2vec_model(sentences, vector_size, window, epochs):
    start_time = time.time()
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=1, workers=4, sg=1, epochs=epochs)
    end_time = time.time()
    training_time = end_time - start_time
    return model, training_time


def calculate_normalized_centroids(model, labelled_transactions):
    print("Calculating centroids...")
    start_time = time.time()
    dim = model.wv[labelled_transactions[0][0]].shape[0]
    num_labels = len(labelled_transactions)
    centroids = np.zeros((num_labels, dim))
    for i,transaction in enumerate(tqdm(labelled_transactions)):
        word_vectors = np.array([model.wv   [it] for it in transaction if it in model.wv])
        if word_vectors.size > 0:
            mean_vector = np.mean(word_vectors, axis=0)
            centroids[i] = normalize(mean_vector.reshape(1, -1), norm='l2').reshape(dim,)
    
    end_time = time.time()
    training_time = end_time - start_time
    return centroids, training_time



def w2v(file_path, vector_size, window, epochs):
    

    transactions, classes= load_data(file_path)


    print("starting training...")
    model_dw, training_time = train_word2vec_model(transactions, vector_size=vector_size, window=window, epochs=epochs)

    print(f"Model trained in {training_time:.2f} seconds")

    X, centroide_time=calculate_normalized_centroids(model_dw, transactions)
    del transactions
    del model_dw
    X = X.astype(np.float32)

    output_file = f"../../../datasets/w2v_transactions/{os.path.basename(file_path)}_{vector_size}_{window}_{epochs}_{training_time+centroide_time}.data"

    with open(output_file, "w", encoding="utf-8") as out:
        chunk_size=100000
        buffer = []
        for i, (cls, text) in enumerate(tqdm(zip(classes, X))):  
            text_str = " ".join(map(str, text))  
            buffer.append(f"{cls};{text_str}\n") 
            
            if (i + 1) % chunk_size == 0:
                out.writelines(buffer)
                buffer = []
        if buffer:
            out.writelines(buffer)








def main():
    print("Starting W2V...")
    dataset_folder = "../../../datasets/dataset_db_w2v"
    
    # Define the settings for window, vector size, epochs, and n_clusters
    # window_settings = [5, 10]
    # vector_size_settings = [50, 100, 200, 400]
    # epochs_settings = [5, 10, 20]
    window_settings = [5]
    vector_size_settings = [200]
    epochs_settings = [10]

    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data'):
            settings_combinations = product(window_settings, vector_size_settings, epochs_settings)
            file_path = os.path.join(dataset_folder, filename)
            for window, vector_size, epochs in settings_combinations:
                w2v(file_path, vector_size, window, epochs)
              
            

if __name__ == "__main__":
    main()

    