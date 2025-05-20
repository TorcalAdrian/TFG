import numpy as np
import faiss
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import os
import csv

# media vaianza maxima
def load_data_db(file_path,class_path):
    classes = []
    transactions = []
    
    with open(class_path, mode='rt', encoding='UTF-8') as file:
        next(file)
        for line in file:
            parts = line.strip().split(';')
            class_ = parts[1]
            classes.append(class_)
    
    transactions = np.load(file_path)


    return classes, transactions


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


def calcular_elbow(K_range, inertia):

    points = np.array(list(zip(K_range, inertia)))
    start = points[0]
    end = points[-1]
    line_vec = end - start
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    vec_from_start = points - start
    scalar_proj = np.dot(vec_from_start, line_vec_norm)
    proj = np.outer(scalar_proj, line_vec_norm)
    dist_to_line = np.linalg.norm(vec_from_start - proj, axis=1)
    elbow_idx = np.argmax(dist_to_line)
    return K_range[elbow_idx], inertia[elbow_idx]





def run_kmeans_clustering(file_path,vector_size, window, epochs,i):

    K_range = list(range(4, 151, 4)) + list(range(200, 701, 50))
    inertia = []
    iterations = 20
    basename = os.path.basename(file_path)
    basename = os.path.splitext(basename)[0]

    
    output_csv = '../../../final_results/k-elbow/k_elbow_resume.csv'


    file_exists = os.path.isfile(output_csv)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(['filename', 'elbow_k', 'elbow_inertia','nmi','elbow_time', 'iteration'])


        print(f"Iteration {i+1}/{iterations}")
        classes, X = load_data(file_path)
        model_dw, training_time = train_word2vec_model(X, vector_size=vector_size, window=window, epochs=epochs)
        X,embeding_time=calculate_normalized_centroids(model_dw, X)
        dim=X.shape[1]
        X = X.astype(np.float32)
        start_time = time.time()
        for k in tqdm(K_range):

            kmeans = faiss.Kmeans(d=dim, k=k, niter=1, verbose=False)
            kmeans.train(X)
            D, I = kmeans.index.search(X, 1)
            inertia_value = np.sum(D)
            inertia.append(inertia_value)
            
        elbow_k,elbow_inertia = calcular_elbow(K_range, inertia)
        end_time = time.time()
        elbow_time = end_time - start_time
        kmeans = faiss.Kmeans(d=dim, k=elbow_k, niter=1, verbose=False)
        kmeans.train(X)
        y_kmeans = kmeans.index.search(X.astype(np.float32), 1)[1].flatten()
        nmi = normalized_mutual_info_score(np.ravel(classes), y_kmeans)

        plt.figure(figsize=(18, 9))  
        plt.plot(K_range, inertia, 'bo-')
        plt.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow k = {elbow_k}')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia')
        plt.title(f'K-Elbow Method para {basename}')
        plt.grid(True)


        path=f'../../../final_results/k-elbow/{basename}_v2/'
        if not os.path.exists(path):
            os.makedirs(path)
        output_file = f'{path}/{basename}_{i+1}.png'
        plt.savefig(output_file, format='png')

        plt.close()
        writer.writerow([basename, elbow_k, elbow_inertia,nmi,elbow_time,i+1])



def main():
    dataset_folder = "../../../datasets/final_datasets/"
    
    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data')  and filename!= "db2014-sep_w2v.data" and filename!= "db201610-sep_w2v.data":
            file_path = os.path.join(dataset_folder, filename)
            for i in range(20):
                run_kmeans_clustering(file_path, 200, 5, 5,i)
              




              
            

if __name__ == "__main__":
    main()           
