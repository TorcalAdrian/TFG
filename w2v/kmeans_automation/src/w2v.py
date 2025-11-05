import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import datetime
from tabulate import tabulate
import os
import time

from tqdm import tqdm


def load_data(file_path):
    transactions = []
    with open(file_path, mode='rt', encoding='UTF-8') as file:
        next(file)
        for line in file:
            # i+=1
            # if i==10:
            #     break
            parts = line.strip().split(';')
            events = parts[2].split()
            transactions.append(events)

    return transactions

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


    transactions= load_data(file_path)


    print("starting training...")
    model_dw, training_time = train_word2vec_model(transactions, vector_size=vector_size, window=window, epochs=epochs)




    X, centroide_time=calculate_normalized_centroids(model_dw, transactions)
    del transactions
    del model_dw
    X = X.astype(np.float32)


    output_file = f"../../../datasets/w2v_temp_files/{os.path.basename(file_path).replace('.data','')}.npy"
    np.save(output_file, X)
    return training_time, centroide_time, output_file









def get_transactions(filename, dataset_folder, window, vector_size, epochs):
    training_time = 0
    centroide_time = 0
    output_file = ""

    print("Extracting transactions from file:", filename)
    file_path = os.path.join(dataset_folder, filename)
    training_time,centroide_time, output_file = w2v(file_path, vector_size, window, epochs)


    return training_time, centroide_time,output_file





