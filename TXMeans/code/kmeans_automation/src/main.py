import os
from itertools import product
from kmeans_clustering import run_kmeans_clustering

def main():
    dataset_folder = "../datasets"

    # Define the settings for window, vector size, epochs, and n_clusters
    window_settings = [5, 10, 15]
    vector_size_settings = [50, 100, 150, 200, 300, 500]
    epochs_settings = [5, 10, 20, 50]
    n_clusters_settings = [2, 4, 8, 10, 16, 20, 50, 100]

    settings_combinations = product(window_settings, vector_size_settings, epochs_settings, n_clusters_settings)



    for filename in os.listdir(dataset_folder):
        if filename.endswith('.data'):
            file_path = os.path.join(dataset_folder, filename)
            for window, vector_size, epochs, n_clusters in settings_combinations:
                run_kmeans_clustering(file_path, n_clusters, vector_size, window, epochs)
            print(f"Processing file: {filename}")

if __name__ == "__main__":
    main()