import pandas as pd
import numpy as np

def load_data(file_path, sep):
    df = pd.read_csv(file_path, header=None, sep=sep, engine='python')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def create_class(filename_data, filename_class):
    df_data = load_data(filename_data, sep='::')
    unique_values = df_data[1].unique()
    unique_df = pd.DataFrame(unique_values, columns=['Unique_Values'])
    unique_df.to_csv(filename_class, index=False, header=False)

def process_data(filename_data, filename_class):
    df_class=load_data(filename_class, sep=';')
    df_data=load_data(filename_data, sep='::')


    class_mapping = {row[0]: idx for idx, row in df_class.iterrows()}

    df_data[1] = df_data[1].map(class_mapping)
    

    df_data = df_data[[1, 0]]

    output_filename_tx = filename_data.replace('.dat', '_w2v.dat')
    df_data.to_csv(output_filename_tx, index=True, header=False, sep=';')
    



def add_column_names(filename_data, column_names, sep=';'):
    # Cargar los datos
    df = load_data(filename_data, sep)
    
    # Asignar nombres a las columnas
    df.columns = column_names
    
    # Guardar el DataFrame en un archivo con la nueva cabecera
    output_filename = filename_data
    df.to_csv(output_filename, index=False, header=True, sep=sep)
    print(f"Archivo guardado con cabecera: {output_filename}")

column_names = ['ID', 'CLASS', 'EVENTS']

path = '../dataset_db/'
dataset_class = 'typeSets-db2201610_new.dat'
dataset_data = 'db201610-sep.dat'
filename_class = path + dataset_class
filename_data = path + dataset_data
# create_class(filename_data, filename_class)
# process_data(filename_data, filename_class)

add_column_names("../dataset_db_w2v/db201610-sep_w2v.dat", column_names)

