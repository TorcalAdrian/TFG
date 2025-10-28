import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# "adult_tx","chessBig_tx","connect_tx","letrecog_tx","pendigits_tx"
# "db36-sep_w2v", "db201610-sep_w2v", "db2014-sep_w2v"
# "P0.O0_tx", "P0.O30_tx", "P20.O0_tx", "P20.O30_tx", "P50.O0_tx", "P50.O30_tx", "T500k.D20k.L50.P60.O40.C8"

# nombres_amigables = {
#     "adult_tx": "Adult",
#     "chessBig_tx": "ChessBig",
#     "connect_tx": "Connect",
#     "letrecog_tx": "Letrecog",
#     "pendigits_tx": "Pendigits"
# }

# nombres_amigables = {
#     "P0.O0_tx": "P0, O0",
#     "P0.O30_tx": "P0, O30",
#     "P20.O0_tx": "P20, O0",
#     "P20.O30_tx": "P20, O30",
#     "P50.O0_tx": "P50, O0",
#     "P50.O30_tx": "P50, O30",
#     "T500k.D20k.L50.P60.O40.C8": "T500k ..."
# }

# nombres_amigables = {
#     "db36-sep_w2v": "DBPedia 3.6",
#     "db2014-sep_w2v": "DBPedia 2014",
#     "db201610-sep_w2v": "DBPedia 2016-10"
# }

# Leer archivos
df_tx = pd.read_csv('txmeans_all_datasets.csv')
df_tx['algoritmo'] = 'TXMeans'

df_tk = pd.read_csv('tkmeans_all_datasets.csv')
df_tk['algoritmo'] = 'TKMeans'

df_w2v = pd.read_csv('w2v_all_datasets.csv')
df_w2v['algoritmo'] = 'W2Vect-KMeans'

# Selección de datasets
datasets_seleccionados = [
    "db36-sep_w2v", "db2014-sep_w2v","db201610-sep_w2v"
]
algoritmos_seleccionados = ['TXMeans', 'W2Vect-KMeans']

# Limpiar nombres de dataset
def limpiar_nombre_dbpedia(nombre):
    if nombre.endswith('X2'):
        return None
    if '.data_' in nombre:
        return nombre.split('.data_')[0]
    return nombre

# Obtener medias y desviaciones 
# estándar
def obtener_tiempos_totales(df):
    tiempos_dict = {}

    df = df[df['iteration'] != 'X2']
    df['dataset'] = df['filename'].apply(limpiar_nombre_dbpedia)
    df = df[df['dataset'].notnull()]

    for (ds, algo), group in df.groupby(['dataset', 'algoritmo']):
        if algo not in algoritmos_seleccionados:
            continue

        if ds not in tiempos_dict:
            tiempos_dict[ds] = {}

        if algo in ['TXMeans', 'TKMeans']:
            times = pd.to_numeric(group['running_time'], errors='coerce').fillna(0)
            tiempos_dict[ds][algo] = {
                'media': times.mean(),
                'std': times.std()
            }

        elif algo == 'W2Vect-KMeans':
            rt = pd.to_numeric(group['running_time'], errors='coerce').fillna(0)
            tt = pd.to_numeric(group['training_time'], errors='coerce').fillna(0)
            et = pd.to_numeric(group['embeding_time'], errors='coerce').fillna(0)
            total = rt + tt + et

            tiempos_dict[ds][algo] = {
                'running_time': rt.mean(),
                'training_time': tt.mean(),
                'embeding_time': et.mean(),
                'total': total.mean(),
                'std': total.std()
            }

    return tiempos_dict

# Unir todos los datos
df = pd.concat([df_tx, df_tk, df_w2v], ignore_index=True)
tiempos_dict = obtener_tiempos_totales(df)

# Completar ausencias con ceros
for ds in tiempos_dict:
    for algo in algoritmos_seleccionados:
        if algo not in tiempos_dict[ds]:
            if algo == 'W2Vect-KMeans':
                tiempos_dict[ds][algo] = {
                    'running_time': 0.0,
                    'training_time': 0.0,
                    'embeding_time': 0.0,
                    'total': 0.0,
                    'std': 0.0
                }
            else:
                tiempos_dict[ds][algo] = {
                    'media': 0.0,
                    'std': 0.0
                }

# Filtrar datasets a mostrar
tiempos_dict_filtrado = {k: tiempos_dict[k] for k in datasets_seleccionados if k in tiempos_dict}

# --- PLOTEO ---
fig, ax = plt.subplots(figsize=(20, 15))

indices = np.arange(len(tiempos_dict_filtrado))
n_algos = len(algoritmos_seleccionados)
bar_width = 0.8 / n_algos
posiciones = [indices + bar_width * (i - n_algos / 2 + 0.5) for i in range(n_algos)]

for i, algo in enumerate(algoritmos_seleccionados):
    if algo in ['TXMeans', 'TKMeans']:
        tiempos = [tiempos_dict_filtrado[ds][algo]['media'] for ds in tiempos_dict_filtrado]
        stds = [tiempos_dict_filtrado[ds][algo]['std'] for ds in tiempos_dict_filtrado]

        ax.bar(posiciones[i], tiempos, bar_width, label=algo)
        ax.errorbar(posiciones[i], tiempos, yerr=stds, fmt='none', capsize=5, ecolor='black', elinewidth=1)

        for j, val in enumerate(tiempos):
            if val > 0:
                ax.text(
                    posiciones[i][j],
                    val + 0.5,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=60, fontweight='bold'
                )

    elif algo == 'W2Vect-KMeans':
        running = [tiempos_dict_filtrado[ds][algo]['running_time'] for ds in tiempos_dict_filtrado]
        training = [tiempos_dict_filtrado[ds][algo]['training_time'] for ds in tiempos_dict_filtrado]
        embedding = [tiempos_dict_filtrado[ds][algo]['embeding_time'] for ds in tiempos_dict_filtrado]
        total = [tiempos_dict_filtrado[ds][algo]['total'] for ds in tiempos_dict_filtrado]
        stds = [tiempos_dict_filtrado[ds][algo]['std'] for ds in tiempos_dict_filtrado]

        bars_run = ax.bar(posiciones[i], running, bar_width, label=f'{algo} Clustering Time')
        bars_train = ax.bar(posiciones[i], training, bar_width, bottom=running, label=f'{algo} Training Time')
        bottoms = np.array(running) + np.array(training)
        bars_embed = ax.bar(posiciones[i], embedding, bar_width, bottom=bottoms, label=f'{algo} Embedding Time')

        # Error bars
        ax.errorbar(posiciones[i], total, yerr=stds, fmt='none', capsize=5, ecolor='black', elinewidth=1)

        for j, val in enumerate(total):
            if val > 0:
                ax.text(
                    posiciones[i][j],
                    val + 0.5,
                    f'{val:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=60,
                    fontweight='bold'
                )

# Ajustes finales
nombres_amigables = {
    "db36-sep_w2v": "DBPedia 3.6",
    "db2014-sep_w2v": "DBPedia 2014",
    "db201610-sep_w2v": "DBPedia 2016-10"
}

etiquetas_x = [nombres_amigables.get(ds, ds) for ds in tiempos_dict_filtrado.keys()]
ax.set_xticks(indices)  # <-- Asegura posición de ticks
ax.set_xticklabels(etiquetas_x, rotation=23, ha='right', fontsize=60)



ax.set_ylabel('Tiempo total (segundos)',fontsize=60)
ax.set_title('Comparación de tiempos por dataset y algoritmo',fontsize=60, weight='bold')

ax.legend(title="",
          title_fontsize=28,
          fontsize=38,
          loc='upper center',
          bbox_to_anchor=(0.5, -0.25),  # más abajo que antes
          ncol=2,
          frameon=True)

# plt.subplots_adjust(bottom=0.2)  # margen inferior más grande


plt.xlabel("") 
plt.tight_layout()
plt.yticks(fontsize=45)
plt.show()
