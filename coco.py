""" COLORES                     
rojo: machos adultos            1   [0, 0, 100] - [50, 50, 255] radio=10
magenta: machos subadultos      2   [200, 0, 200], [255, 100, 255] 
marrón/café: hembras adultas    3   [10, 20, 100], [40, 70, 150]
azul: juveniles                 4   [100, 0, 0], [255, 150, 100]
verde: cachorros                5   [0, 100, 0], [100, 255, 100]
"""

import numpy as np
from scipy.ndimage import correlate

def correlacion_cruzada_prototipos(img1, img2):
    # Asegurarse de que ambas imágenes tienen las mismas dimensiones
    if img1.shape != img2.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones")

    # Convertir imágenes a matrices de flotantes
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    # Restar la media para cada imagen
    img1_mean = img1.mean()
    img2_mean = img2.mean()
    img1 = img1 - img1_mean
    img2 = img2 - img2_mean
    
    # Calcular la correlación cruzada
    numerador = np.sum(img1 * img2)
    denominador = np.sqrt(np.sum(img1 ** 2) * np.sum(img2 ** 2))
    correlacion = numerador / denominador
    
    # Convertir a porcentaje
    porcentaje_correlacion = correlacion * 100
    
    return porcentaje_correlacion

def actualizar_lista_prototipos(lista, c, R):
    # Convertir c a arreglo de NumPy si no lo es
    if not isinstance(c, np.ndarray):
        c = np.array(c)
    
    lista_actualizada = []
    agregado = False
    
    for p in lista:
        # Calcular la correlación cruzada entre p y c
        corr = correlacion_cruzada_prototipos(p, c)
        
        if corr > R:
            # Combinar p y c obteniendo la media
            print(f"prototipo {c[0][0]} mezclado con {p[0][0]}")
            media = (p + c) / 2.0
            lista_actualizada.append(media)
            agregado = True
        else:
            lista_actualizada.append(p)
    
    # Si no se agregó c a la lista, añadirlo como un nuevo elemento
    if not agregado:
        print("prototipo no mezclado")
        lista_actualizada.append(c)
    
    return lista_actualizada

def filtrar_coordenadas(coordenadas, radio):
    # Convertir las coordenadas a un array de NumPy
    coords = np.array(coordenadas)
    
    # Lista para almacenar las coordenadas filtradas
    coordenadas_filtradas = []
    cordenadas_limpias=[]
    
    while len(coords) > 0:
        # Tomar la primera coordenada y añadirla a las filtradas
        coord = coords[0]
        coordenadas_filtradas.append(coord)
        
        # Calcular la distancia euclidiana desde la coordenada actual a todas las demás
        distancias = np.sqrt(np.sum((coords - coord) ** 2, axis=1))
        
        # Filtrar las coordenadas que están fuera del radio
        coords = coords[distancias > radio]

        coord_casi_listas=np.array(coordenadas_filtradas).tolist()
        for coord in coord_casi_listas:
            cordenadas_limpias.append([coord[0],coord[1]])

    return cordenadas_limpias


def correlacion_cruzada_normalizada(img, prototipo):
    # Asegurarse de que ambas matrices tienen las mismas dimensiones
    if img.shape[0] < prototipo.shape[0] or img.shape[1] < prototipo.shape[1]:
        raise ValueError("La imagen debe ser más grande que el prototipo")

    # Restar la media para cada imagen
    img_mean = img.mean()
    prototipo_mean = prototipo.mean()
    img = img - img_mean
    prototipo = prototipo - prototipo_mean
    
    # Calcular las desviaciones estándar
    img_std = np.std(img)
    prototipo_std = np.std(prototipo)
    
    # Normalizar las imágenes
    if img_std == 0 or prototipo_std == 0:
        return np.zeros(img.shape)
    
    img = img / img_std
    prototipo = prototipo / prototipo_std

    # Calcular la correlación cruzada normalizada
    correlacion = correlate(img, prototipo, mode='constant', cval=0.0)
    
    return correlacion / (img.shape[0] * img.shape[1] * prototipo_std * img_std)

def croco_buscar_nuevos(prototipo, radio, img, R,lista_corr):
    print("Calculando correlación cruzada normalizada")
    correlacion = correlacion_cruzada_normalizada(img, prototipo)
    lista_corr.append(np.max(correlacion))
    print(np.max(correlacion))
    print("Buscando coordenadas")
    coordenadas = np.argwhere(correlacion > R)
    print(coordenadas)
    cordenadas_limpias=[]
    for coord in coordenadas:
        cordenadas_limpias.append([coord[0],coord[1]])
    print(cordenadas_limpias)
    # Filtrar coordenadas para evitar puntos cercanos
    print(f"Filtrando {len(coordenadas)} coordenadas ")
    coordenadas_filtradas = filtrar_coordenadas(coordenadas, radio)
    
    return coordenadas_filtradas