import os
from itertools import product
from kmeans_clustering import run_kmeans_clustering
from w2v import w2v

def main():
    dataset_folder = "../../../dataset_db_w2v"
    
    # Define the settings for window, vector size, epochs, and n_clusters
    # window_settings = [5, 10]
    # vector_size_settings = [50, 100, 200, 400]
    # epochs_settings = [5, 10, 20]
    window_settings = [5]
    vector_size_settings = [200]
    epochs_settings = [5]


    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data') and filename!= "db36-sep_w2v.data":
            settings_combinations = product(window_settings, vector_size_settings, epochs_settings)
            file_path = os.path.join(dataset_folder, filename)
            for window, vector_size, epochs in settings_combinations:
                run_kmeans_clustering(file_path, vector_size, window, epochs)
              
            

if __name__ == "__main__":
    main()