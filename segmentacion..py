import cv2
import numpy as np


# Cargar la imagen
imagen = cv2.imread('img/Train/41.jpg')

# Verificar si la imagen se cargó correctamente
if imagen is None:
    print("Error: No se pudo cargar la imagen.")
    exit(1)

# Convertir la imagen a RGB (OpenCV carga imágenes en BGR por defecto)
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

# Definir el rango de colores de interés en formato RGB
# Por ejemplo, para segmentar el color azul
lower_blue = np.array([87, 80, 71], dtype=np.uint8)
upper_blue = np.array([237, 230, 219], dtype=np.uint8)

# Crear una máscara con los píxeles que estén dentro del rango especificado
mascara = cv2.inRange(imagen_rgb, lower_blue, upper_blue)

# Aplicar la máscara a la imagen original para obtener la imagen segmentada
imagen_segmentada = cv2.bitwise_and(imagen_rgb, imagen_rgb, mask=mascara)

# Redimensionar la imagen a una anchura de 500 píxeles, ajustando automáticamente la altura
anchura_deseada = 1000
altura_original, anchura_original, _ = imagen_segmentada.shape
nueva_altura = int(altura_original * (anchura_deseada / anchura_original))
imagen_redimensionada = cv2.resize(imagen_segmentada, (anchura_deseada, nueva_altura))

# Convertir las imágenes de nuevo a BGR para la visualización correcta con OpenCV
imagen_rgb_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)
imagen_segmentada_bgr = cv2.cvtColor(imagen_segmentada, cv2.COLOR_RGB2BGR)
imagen_redimensionada_bgr = cv2.cvtColor(imagen_redimensionada, cv2.COLOR_RGB2BGR)

# Mostrar la imagen original y la imagen segmentada
cv2.imshow('Imagen Original', imagen_rgb_bgr)
cv2.imshow('Imagen Segmentada', imagen_segmentada_bgr)
cv2.imshow('Imagen Segmentada Redimensionada', imagen_redimensionada_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
