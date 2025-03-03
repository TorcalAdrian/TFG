import os
import pandas as pd

def get_top_30_executions(file_path):
    df = pd.read_csv(file_path)
    top_30 = df.nlargest(30, 'NMI')
    return top_30

def main():
    results_folder = '../results_raw'
    resume_folder = '../results_resume'
    
    for file_name in os.listdir(results_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(results_folder, file_name)
            top_30 = get_top_30_executions(file_path)
            output_file = os.path.join(resume_folder, f'resume_{file_name}')
            top_30.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()