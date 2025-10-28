import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg") 
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("TkAgg")

# Leer y unir datos
df_tx = pd.read_csv('txmeans_all_datasets.csv')
df_tx['algoritmo'] = 'TXMeans'

df_tk = pd.read_csv('tkmeans_all_datasets.csv')
df_tk['algoritmo'] = 'TKMeans'

df_w2v = pd.read_csv('w2v_txmeans_cluster.csv')
df_w2v['algoritmo'] = 'W2Vect-KMeans'
df_w2v = pd.read_csv('w2v_all_datasets.csv')
df_w2v['algoritmo'] = 'W2Vect-KMeans'

df = pd.concat([df_tx,df_tk,df_w2v], ignore_index=True)
df['nmi'] = pd.to_numeric(df['nmi'], errors='coerce')
df = df.dropna(subset=['nmi'])
df = df[~df['filename'].str.strip().str.endswith('X2')]

def limpiar_nombre_dbpedia(nombre):
    if '.data_' in nombre:
        nombre = nombre.split('.data_')[0]
    return nombre.strip()

df['filename_limpio'] = df['filename'].apply(limpiar_nombre_dbpedia)

# Muy importante: filtrar los algoritmos que tienen ncluster (TKMeans y W2Vect-KMeans)
df_filtrado = pd.concat([
    df[df['algoritmo'] == 'TXMeans'],  # TXMeans sin filtrar
    df[(df['algoritmo'] != 'TXMeans') & (df['ncluster'] == '16')]  # TKMeans y W2Vect-KMeans filtrados
])
# df_filtrado=df
# Selección de datasets
# Diccionario para renombrar los datasets en el eje X
nombres_amigables = {
    "P0.O0_tx": "P0, O0",
    "P0.O30_tx": "P0, O30",
    "P20.O0_tx": "P20, O0",
    "P20.O30_tx": "P20, O30",
    "P50.O0_tx": "P50, O0",
    "P50.O30_tx": "P50, O30",
    "T500k.D20k.L50.P60.O40.C8": "T500k ..."
}


datasets_seleccionados = ["P0.O0_tx", "P0.O30_tx", "P20.O0_tx", "P20.O30_tx", "P50.O0_tx", "P50.O30_tx", "T500k.D20k.L50.P60.O40.C8"]
algoritmos = ['TXMeans','TKMeans','W2Vect-KMeans']


# Calcular media y desviación estándar
resultados = []
for dataset in datasets_seleccionados:
    for algoritmo in algoritmos:
        subset = df_filtrado[(df_filtrado['filename_limpio'] == dataset) & (df_filtrado['algoritmo'] == algoritmo)]
        ejecuciones = subset[subset['iteration'] != 'avg']['nmi']
        avg_row = subset[subset['iteration'] == 'avg']['nmi']
        media = avg_row.values[0] if not avg_row.empty else ejecuciones.mean()
        std = ejecuciones.std() if not ejecuciones.empty else 0
        resultados.append({'dataset': dataset, 'algoritmo': algoritmo, 'media': media, 'std': std})

df_plot = pd.DataFrame(resultados)
# Reemplazar nombres en df_plot
df_plot['dataset_legible'] = df_plot['dataset'].map(nombres_amigables)
# Colores personalizados
colores = {
    'TXMeans': '#1f77b4',
    'TKMeans': '#2ca02c',
    'W2Vect-KMeans': '#ff7f0e'
}

# Plot
sns.set(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(35, 16), constrained_layout=True)

ax = sns.barplot(data=df_plot, x='dataset_legible', y='media', hue='algoritmo', palette=colores, ci=None, dodge=True)


# Etiquetas de medias encima de cada barra
# Etiquetas de medias encima de cada barra
for bars in ax.containers:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 20),  # Más alto
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontsize=43,      # Más grande
                    fontweight='bold')

offsets = {
    'TXMeans': -0.2,
    'W2Vect-KMeans': 0.2
}
# Añadir barras de error
for i, row in df_plot.iterrows():
    pos = list(df_plot['dataset_legible'].unique()).index(row['dataset_legible']) + offsets.get(row['algoritmo'], 0)

    ax.errorbar(
        x=pos,
        y=row['media'],
        yerr=row['std'],
        fmt='none',
        ecolor='black',
        elinewidth=3,
        capsize=5,
        capthick=3
    )

plt.title("Comparativa de NMI por Dataset y Algoritmo", fontsize=55, weight='bold',y=1.08)
plt.yticks(fontsize=55)  
plt.ylabel("Normalized Mutual Information (NMI)", fontsize=55)
plt.xticks(rotation=25, ha='right', fontsize=55)
plt.legend(title="",
           title_fontsize=35,
           fontsize=45,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.25),
           ncol=3,
           frameon=True)
plt.xlabel("")  

plt.tight_layout()
plt.show()
