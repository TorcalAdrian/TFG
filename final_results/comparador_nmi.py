import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Carga de los datos
df_tx = pd.read_csv('txmeans.csv')
df_tx['algoritmo'] = 'TXMeans'

df_tk = pd.read_csv('w2v.csv')
df_tk['algoritmo'] = 'W2vect-KMeans'

# Dataset a comparar (ajusta el nombre según necesites)
dataset_comparar = "db201610-sep_w2v"

# Filtrar solo ejecuciones individuales (no 'avg')
df_tx = df_tx[(df_tx['iteration'] != 'avg')]
df_tk = df_tk[(df_tk['iteration'] != 'avg') ]

# Convertir iteration a entero (por si acaso viene como string)
df_tx['iteration'] = df_tx['iteration'].astype(int)
df_tk['iteration'] = df_tk['iteration'].astype(int)

# Ordenar por iteración para asegurar consistencia
df_tx = df_tx.sort_values(by='iteration')
df_tk = df_tk.sort_values(by='iteration')

# # Crear el gráfico
# plt.figure(figsize=(10, 5))

# # Gráfico de líneas o puntos
# plt.plot(df_tx['iteration'], df_tx['nmi'], marker='o', label='TXMeans', color='blue', linewidth=2)
# plt.plot(df_tk['iteration'], df_tk['nmi'], marker='s', label='W2vect-KMeans', color='orange', linewidth=2)

# plt.title(f'Variabilidad del NMI en 20 ejecuciones\nDataset: DBpedia 2016-10')
# plt.xlabel('Ejecución')
# plt.ylabel('NMI')
# plt.xticks(range(0, 20))
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.show()
palette_azul = {
    'TXMeans': '#1f77b4',      # azul clásico
    'TKMeans': '#1f77b4',      # mismo azul para ambos, o puedes cambiar
    'W2Vect-KMeans': '#1f77b4'
}
df_box = pd.concat([df_tx], ignore_index=True)

plt.figure(figsize=(8, 6))
sns.boxplot(x='algoritmo', y='nclusters', data=df_box, palette=palette_azul)
sns.stripplot(x='algoritmo', y='nclusters', data=df_box, 
              color='black', alpha=0.9, jitter=True, size=8)

plt.title('Distribución del número de clusters en 20 ejecuciones (con puntos individuales)')
plt.ylabel('Numero de clusters')
plt.xlabel('Algoritmo')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
