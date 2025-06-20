import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Cargar imagen
img = mpimg.imread('db36-sep_w2v_11_editada.png')

fig, ax = plt.subplots(figsize=(30, 28))

# Mostrar la imagen
ax.imshow(img)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)



# Añadir título (cabecera)
plt.title('K-Elbow Method para db36-sep_w2v', fontsize=28, pad=10, y=0.88)


# Añadir leyenda eje Y
plt.ylabel('Inercia', fontsize=28, labelpad=0, rotation=90)

# Añadir leyenda eje X
plt.xlabel('Número de Clústeres (K)', fontsize=28, labelpad=20)
ax.yaxis.set_label_coords(0.05, 0.5)
# Guardar imagen resultante
plt.savefig('db36-sep_w2v_11_con_cabecera_y_leyendas_matplotlib.png', bbox_inches='tight', dpi=300)

# Mostrar en pantalla (opcional)
plt.show()
