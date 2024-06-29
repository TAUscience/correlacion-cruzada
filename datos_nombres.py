import csv

TRAIN=[
    "42",
    "44",
    "46",
    "48",
    "50",
    "45",
    "49",
    "47",
    "43"
]
TEST=[
    "41"
]

def obtener_etiquetas(valor_inicial, rutacsv):
    diccionario = {}
    with open(rutacsv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Saltar la primera fila si es el encabezado
        for row in reader:
            if row[0].strip() == valor_inicial:  # Comprobar si el primer elemento de la fila es el valor inicial buscado
                # Crear la entrada en el diccionario, usando el primer elemento como clave y el resto como valores en una lista
                diccionario[row[0]] = [int(x) for x in row[1:]]  # Convertir los valores a enteros si es necesario
                break  # Terminar después de encontrar la primera coincidencia
    
    return diccionario