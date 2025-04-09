import os
from txmeans_clustering import run_txmeans_clustering

def main():
    dataset_folder = "../../datasets/dataset_db_w2v"


    for filename in os.listdir(dataset_folder):
        print(f"Processing file: {filename}")
        if filename.endswith('.data'):
            file_path = os.path.join(dataset_folder, filename)
            run_txmeans_clustering(file_path)
            

if __name__ == "__main__":
    main()