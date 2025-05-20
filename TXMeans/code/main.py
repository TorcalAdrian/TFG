import os
from TKmeans_clustering import run_txmeans_clustering

def main():
    dataset_folder = "../../datasets/final_datasets_v2/"


    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data') and filename  in ["db36-sep_w2v.data"]:
            file_path = os.path.join(dataset_folder, filename)
            run_txmeans_clustering(file_path)
            

if __name__ == "__main__":
    main()