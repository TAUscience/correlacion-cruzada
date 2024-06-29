import numpy as np
import cv2

import coco
import segmentacion as seg
import datos_nombres as dt

img_train=[]
img_train_dot=[]
img_test=[]
labels_test={}
etiquetas_color={
    "0":[[0, 0, 100], [50, 50, 255], 20],
    "1":[[200, 0, 200], [255, 100, 255],15],
    "2":[[15, 25, 105], [40, 70, 150],16],
    "3":[[151, 48, 24], [180, 60, 50],10],
    "4":[[20, 160, 30], [40, 180, 60],7]
}
coordenadas={
}
prototipos={}

#Obtener img de entrenamiento redimensionadas
for nombre in dt.TRAIN:
    print(nombre)
    ruta=f"img/Train/{nombre}.jpg"
    ruta_dot=f"img/TrainDotted/{nombre}.jpg"
    img_train.append(seg.redim(ruta,2000))
    img_train_dot.append(seg.redim(ruta_dot,2000))

#Obtener img de prueba redimensionadas y sus resultados
for nombre in dt.TEST:
    print(nombre)
    ruta=f"img/Train/{nombre}.jpg"
    img_test.append(seg.redim(ruta,2000))
    labels_test.update(dt.obtener_etiquetas(nombre,"img/Train/train.csv"))

#Obtener los prototipos ENTRENAMIENTO
croco_proto=5 #Correlación entre prototipos
for clase in range(5):
    clave=f"{clase}"
    print(clave)
    lista_prototipos=list()
    taman_proto=1+etiquetas_color[clave][2]*2
    for img in range(len(img_train)):
        encontrados = seg.encontrar_puntos_color(img_train_dot[img],etiquetas_color[clave][0],etiquetas_color[clave][1])
        for (y, x) in encontrados:
            prototipo=seg.obtener_subarreglo(img_train[img],y,x,etiquetas_color[clave][2])
            if len(lista_prototipos)==0:
                lista_prototipos.append(prototipo)
            else:
                if (prototipo.shape)[0]!=taman_proto or (prototipo.shape)[1]!=taman_proto:
                    prototipo=lista_prototipos[0]
                else:
                    lista_prototipos=coco.actualizar_lista_prototipos(lista_prototipos,prototipo,croco_proto)
    prototipos[clave]=lista_prototipos

#Buscar leones en las imagenes nuevas
R=0.0000006 #Porcentaje de correlación para decidir clase
for img in range(len(img_test)):
    print(f"Buscando en img test {img}")
    img_etiquetada=np.copy(img_test[img])
    for clase in range(1):
        print(f"Clase: {clase}")
        coord_clase=[]
        clave=f"{clase}"
        i=0
        for proto in prototipos[clave]:
            i+=1
            print(f"Protipo {i}/{len(prototipos[clave])}")
            coord_clase.extend(coco.croco_buscar_nuevos(prototipos[clave][proto], etiquetas_color[clave][2], img_test[img], R))
            print(coord_clase)
        print("Filtrando coordenadas de clase")
        coord_clase=coco.filtrar_coordenadas(coord_clase, etiquetas_color[clave][2])
        print(f"Colocando etiquetas {len(coord_clase)}")
        img_etiquetada=seg.agregar_cuadros_envolventes(coord_clase, img_etiquetada, etiquetas_color[clave][1], etiquetas_color[clave][2])
    # Guardar la imagen con los cuadros envolventes
    cv2.imwrite(f"imagen_con_cuadros{img}.png", img_etiquetada)

