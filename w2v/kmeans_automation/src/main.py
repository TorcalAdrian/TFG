import os
from itertools import product
from kmeans_clustering import run_kmeans_clustering

def main():
    dataset_folder = "../../dataset_v2"

    # Define the settings for window, vector size, epochs, and n_clusters
    window_settings = [5, 10]
    vector_size_settings = [50, 100, 200, 400]
    epochs_settings = [5, 10]

    for filename in os.listdir(dataset_folder):
        if filename.endswith('.data'):
            settings_combinations = product(window_settings, vector_size_settings, epochs_settings)
            file_path = os.path.join(dataset_folder, filename)
            for window, vector_size, epochs in settings_combinations:
                run_kmeans_clustering(file_path, vector_size, window, epochs)
            print(f"Processing file: {filename}")

if __name__ == "__main__":
    main()