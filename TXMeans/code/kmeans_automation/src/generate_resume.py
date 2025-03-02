import os
import pandas as pd

def get_top_30_executions(file_path):
    df = pd.read_csv(file_path)
    top_30 = df.nlargest(30, 'NMI')
    return top_30

def main():
    results_folder = '../results'
    summary_file = os.path.join(results_folder, 'resume.csv')
    
    all_top_30 = []
    
    for file_name in os.listdir(results_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(results_folder, file_name)
            top_30 = get_top_30_executions(file_path)
            all_top_30.append(top_30)
    
    summary_df = pd.concat(all_top_30)
    summary_df.to_csv(summary_file, index=False)

if __name__ == "__main__":
    main()