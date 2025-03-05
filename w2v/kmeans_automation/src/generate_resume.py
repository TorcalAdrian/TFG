import os
import pandas as pd

def get_top_30_executions(file_path):
    df = pd.read_csv(file_path)
    df_median = df[df['iteration'] == 'median']
    top_30 = df_median.nlargest(30, 'NMI').sort_values(by='NMI', ascending=False)
    return top_30

def main():
    results_folder = '../results_raw'
    resume_folder = '../results_resume'
    
    if not os.path.exists(resume_folder):
        os.makedirs(resume_folder)
    
    for file_name in os.listdir(results_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(results_folder, file_name)
            top_30 = get_top_30_executions(file_path)
            output_file = os.path.join(resume_folder, f'resume_{file_name}')
            top_30.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()