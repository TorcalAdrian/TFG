import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Carga de los datos
df_tx = pd.read_csv('txmeans.csv')
df_tx['algoritmo'] = 'TXMeans'

df_tk = pd.read_csv('w2v.csv')
df_tk['algoritmo'] = 'W2Vect-KMeans'

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



palette_azul = {
    'TXMeans': '#1f77b4',      # azul clásico
    'TKMeans': '#1f77b4',      # mismo azul para ambos, o puedes cambiar
    'W2Vect-KMeans': '#1f77b4'
}
df_box = pd.concat([df_tx], ignore_index=True)



plt.figure(figsize=(20, 18))
sns.boxplot(x='algoritmo', y='nclusters', data=df_box, palette=palette_azul)
sns.stripplot(x='algoritmo', y='nclusters', data=df_box, 
              color='black', alpha=0.9, jitter=True, size=23)

plt.title('Distribución del NMI en 20 ejecuciones',fontsize=75)
plt.ylabel('NMI',fontsize=75)
plt.yticks(fontsize=75)  
plt.xlabel('')
plt.xticks( fontsize=75)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
