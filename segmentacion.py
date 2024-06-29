import cv2
import numpy as np
import matplotlib.pyplot as plt

import coco
print("HOLAMUNDO")

def seg_dim(ruta, anchura_deseada):
    # Cargar la imagen
    imagen = cv2.imread(ruta)

    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        exit(1)

    # Convertir la imagen a RGB (OpenCV carga imágenes en BGR por defecto)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    # Definir el rango de colores de interés en formato RGB
    # Por ejemplo, para segmentar el color azul
    lower_blue = np.array([0, 7, 241], dtype=np.uint8) #87,80,71
    upper_blue = np.array([10, 17, 255], dtype=np.uint8) #237,230,219

    # Crear una máscara con los píxeles que estén dentro del rango especificado
    mascara = cv2.inRange(imagen_rgb, lower_blue, upper_blue)

    # Aplicar la máscara a la imagen original para obtener la imagen segmentada
    imagen_segmentada = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=mascara)

    # Redimensionar la imagen a una anchura deseada, ajustando automáticamente la altura
    altura_original, anchura_original, _ = imagen_segmentada.shape
    nueva_altura = int(altura_original * (anchura_deseada / anchura_original))
    imagen_redimensionada = cv2.resize(imagen_segmentada, (anchura_deseada, nueva_altura))

    return imagen_redimensionada #[alto][ancho][canalBGR]


def redim(ruta, anchura_deseada):
    # Cargar la imagen
    imagen = cv2.imread(ruta)

    # Verificar si la imagen se cargó correctamente
    if imagen is None:
        print("Error: No se pudo cargar la imagen.")
        exit(1)

    # Redimensionar la imagen a una anchura deseada, ajustando automáticamente la altura
    altura_original, anchura_original, _ = imagen.shape
    nueva_altura = int(altura_original * (anchura_deseada / anchura_original))
    imagen_redimensionada = cv2.resize(imagen, (anchura_deseada, nueva_altura))

    return np.array(imagen_redimensionada)

def colorear_adyacentes(arreglo_np, b, g, r, radio):
    # Crear una copia de la imagen para modificar
    imagen_modificada = arreglo_np.copy()

    # Obtener las coordenadas de los píxeles que coinciden con el valor BGR dado
    mascara = (arreglo_np[:,:,0] == b) & (arreglo_np[:,:,1] == g) & (arreglo_np[:,:,2] == r)
    coordenadas = np.column_stack(np.where(mascara))

    # Colorear los píxeles adyacentes en un radio dado de color blanco
    for (y, x) in coordenadas:
        cv2.circle(imagen_modificada, (x, y), radio, (255, 255, 255), -1)

    return imagen_modificada

def obt_prototip(arreglo_np, b, g, r):
    fils, cols, _ = arreglo_np.shape
    img_resultado = np.copy(arreglo_np)
    # Definir un kernel de 5x5 para colorear los píxeles adyacentes
    kernel = np.ones((2*5+1, 2*5+1, 3), dtype=np.uint8) * 255

    for f in range(fils):
        for c in range(cols):
            if (arreglo_np[f, c] == [b, g, r]).all():
                print(f,c)
                # Asegurarse de no acceder a índices fuera de los límites
                min_f, max_f = max(f-5, 0), min(f+5+1, fils)
                min_c, max_c = max(c-5, 0), min(c+5+1, cols)
                img_resultado[min_f:max_f, min_c:max_c] = kernel[:max_f-min_f, :max_c-min_c]

    return img_resultado


def encontrar_puntos_color(arreglo_np, lower, upper, distancia_minima=10):
    # Convertir los límites de color a numpy arrays
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    # Crear una máscara para los píxeles que caen dentro del rango de color
    mascara = cv2.inRange(arreglo_np, lower, upper)

    # Obtener las coordenadas de los píxeles dentro del rango de color
    coordenadas = np.column_stack(np.where(mascara > 0))

    # Lista para almacenar las coordenadas filtradas
    coordenadas_filtradas = []

    i=0
    for coord in coordenadas:
        i+=1
        print(f"{i}/{len(coordenadas)}")
        if not coordenadas_filtradas:  # Si la lista está vacía, añadir la primera coordenada
            coordenadas_filtradas.append(coord)
        else:
            distancias = np.linalg.norm(np.array(coordenadas_filtradas) - coord, axis=1)
            if np.all(distancias >= distancia_minima):
                coordenadas_filtradas.append(coord)

    return np.array(coordenadas_filtradas)

def obtener_subarreglo(arreglo_np, y, x, r):
    # Obtener las dimensiones del arreglo
    fils, cols, _ = arreglo_np.shape
    
    # Calcular los límites del subarreglo
    min_y = max(y - r, 0)
    max_y = min(y + r + 1, fils)
    min_x = max(x - r, 0)
    max_x = min(x + r + 1, cols)
    
    # Extraer el subarreglo
    subarreglo = arreglo_np[min_y:max_y, min_x:max_x]

    return subarreglo

def agregar_cuadros_envolventes(coordenadas, imagen, color, radio):
    # Hacer una copia de la imagen para no modificar la original
    imagen_con_cuadros = imagen.copy()
    
    # Convertir el color a una lista si no lo es
    if not isinstance(color, (list, tuple)):
        color = list(color)
    
    # Iterar sobre las coordenadas y agregar los cuadros envolventes
    for coord in coordenadas:
        y, x = coord  # Obtener las coordenadas (y, x)
        
        # Calcular los límites del cuadro envolvente
        y_min = max(y - radio, 0)
        y_max = min(y + radio + 1, imagen.shape[0])
        x_min = max(x - radio, 0)
        x_max = min(x + radio + 1, imagen.shape[1])
        
        # Dibujar el cuadro envolvente
        cv2.rectangle(imagen_con_cuadros, (x_min, y_min), (x_max, y_max), color, 2)
    
    return imagen_con_cuadros

"""
imagen_nueva = redim("img/Train/41.jpg", 3000)
imagen_nueva_dot =redim("img/TrainDotted/41.jpg", 3000)
#encontrados=obt_prototip(imagen_nueva_dot,238,11,7)
encontrados = encontrar_puntos_color(imagen_nueva_dot,[20, 160, 30], [40, 180, 60])
# Mostrar la imagen usando matplotlib
imagen_rgb = cv2.cvtColor(imagen_nueva_dot, cv2.COLOR_BGR2RGB)
plt.imshow(imagen_rgb)
plt.axis('off')  # Ocultar los ejes
plt.show()

# Crear una copia de la imagen para dibujar los puntos
imagen_con_puntos = imagen_nueva_dot.copy()


lista_prototipos=list()
for (y, x) in encontrados:
    cv2.circle(imagen_con_puntos, (x, y), 5, (255, 0, 0), -1)
    prototipo=obtener_subarreglo(imagen_nueva,y,x,10)
    if len(lista_prototipos)==0:
        lista_prototipos.append(prototipo)
    else:
        lista_prototipos=coco.actualizar_lista_prototipos(lista_prototipos,prototipo,30)

# Mostrar la imagen usando matplotlib
imagen_rgb = cv2.cvtColor(imagen_con_puntos, cv2.COLOR_BGR2RGB)
plt.imshow(imagen_rgb)
plt.axis('off')  # Ocultar los ejes
plt.show()
print(len(lista_prototipos))
"""