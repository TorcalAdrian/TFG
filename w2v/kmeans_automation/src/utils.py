import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
import os

def load_dataset(file_path):
    df = pd.read_csv(file_path, sep=';')
    df = df.drop(columns=['ID'])
    return df

def preprocess_text(df, columns_to_use):
    df['processed_text'] = df[columns_to_use].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    df['processed_text'] = df['processed_text'].apply(lambda x: x.split())
    return df

def get_average_vector(words, model_dw):
    vectors = []
    for word in words:
        if word in model_dw.wv:
            vectors.append(model_dw.wv[word])
    if vectors:
        return normalize(np.mean(vectors, axis=0).reshape(1, -1))[0]
    else:
        return np.zeros(model_dw.vector_size)

def load_all_datasets(dataset_folder):
    datasets = []
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.data'):
            file_path = os.path.join(dataset_folder, filename)
            datasets.append(load_dataset(file_path))
    return datasets