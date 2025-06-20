import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Leer y unir datos
df_tx = pd.read_csv('txmeans_all_datasets.csv')
df_tx['algoritmo'] = 'TXMeans'

df_tk = pd.read_csv('tkmeans_all_datasets.csv')
df_tk['algoritmo'] = 'TKMeans'

# df_w2v = pd.read_csv('w2v_txmeans_cluster.csv')
# df_w2v['algoritmo'] = 'W2Vect-KMeans'
df_w2v = pd.read_csv('w2v_all_datasets.csv')
df_w2v['algoritmo'] = 'W2Vect-KMeans'

df = pd.concat([df_tx,df_w2v], ignore_index=True)
df['nmi'] = pd.to_numeric(df['nmi'], errors='coerce')
df = df.dropna(subset=['nmi'])
df = df[~df['filename'].str.strip().str.endswith('X2')]

def limpiar_nombre_dbpedia(nombre):
    if '.data_' in nombre:
        nombre = nombre.split('.data_')[0]
    return nombre.strip()

df['filename_limpio'] = df['filename'].apply(limpiar_nombre_dbpedia)

# Muy importante: filtrar los algoritmos que tienen ncluster (TKMeans y W2Vect-KMeans)
# df_filtrado = pd.concat([
#     df[df['algoritmo'] == 'TXMeans'],  # TXMeans sin filtrar
#     df[(df['algoritmo'] != 'TXMeans') & (df['ncluster'] == '16')]  # TKMeans y W2Vect-KMeans filtrados
# ])
df_filtrado=df
# Selección de datasets
# Diccionario para renombrar los datasets en el eje X
nombres_amigables = {
    "db36-sep_w2v": "DBPedia 3.6",
    "db2014-sep_w2v": "DBPedia 2014",
    "db201610-sep_w2v": "DBPedia 2016-10"
}


datasets_seleccionados = ["db36-sep_w2v", "db201610-sep_w2v", "db2014-sep_w2v"]
algoritmos = ['TXMeans','W2Vect-KMeans']


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
plt.figure(figsize=(23, 13), constrained_layout=True)

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
                    fontsize=34,      # Más grande
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
        elinewidth=1.5,
        capsize=5,
        capthick=1.5
    )

plt.title("Comparativa de NMI por Dataset y Algoritmo", fontsize=35, weight='bold',y=1.08)
plt.yticks(fontsize=35)  
plt.ylabel("Normalized Mutual Information (NMI)", fontsize=35)
plt.xticks(rotation=30, ha='right', fontsize=35)
plt.legend(title="",
           title_fontsize=28,
           fontsize=28,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.25),
           ncol=3,
           frameon=True)
plt.xlabel("")  

plt.tight_layout()
plt.show()
