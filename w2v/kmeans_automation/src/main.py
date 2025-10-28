import os
from itertools import product
from kmeans_clustering import run_kmeans_clustering
# from w2v import w2v

def main():
    dataset_folder = "../../../datasets/final_datasets/"
    


    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data') and filename!= "db36-sep_w2v.data" and filename!= "db2014-sep_w2v.data" and filename!= "db201610-sep_w2v.data" :
            file_path = os.path.join(dataset_folder, filename)
            run_kmeans_clustering(file_path, 200, 5, 5)
            # settings_combinations = product(window_settings, vector_size_settings, epochs_settings)

            # for window, vector_size, epochs in settings_combinations:
            #     run_kmeans_clustering(file_path, vector_size, window, epochs)
              
            

if __name__ == "__main__":
    main()