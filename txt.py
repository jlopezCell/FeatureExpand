import pandas as pd
import numpy as np
from featureexpand.feature_expander import FeatureExpander, encode, generate_variable_map
import json

def generate_json_from_matrix(X,y=None,numberVariables=1):
    # Convertir cada fila de la matriz X en un mintermin (cadena binaria)
    
    mintermins = []
    for i in range(len(X)):
        rx = ""
        for j in range(len(X[i])):
            rx += "".join(map(str, encode(X[i][j],numberVariables)))
        mintermins.append(rx)
    
    print("Minterm", mintermins)
    # Asignar los Cluster en función del primer valor de cada fila
    
    # Crear el diccionario con los datos
    data = {
        "mintermins": mintermins,
        "Cluster": y
    }
    
    # Convertir el diccionario a JSON
    json_data = json.dumps(data, indent=4)
    
    # Guardar el JSON en un archivo (opcional)
    with open("data_from_matrix.json", "w") as json_file:
        json_file.write(json_data)
    
    return json_data

# Llamar a la función para generar el JSON
X = [
    [0, 0.9],
    [0, 1],
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0],
    [1, 1],
    [0, 0]
]

Cluster = ["k0" if row[0] == 0 else "k1" for row in X]
json_output = generate_json_from_matrix(X,Cluster)

print(json_output)

# Crear un DataFrame de ejemplo
data = {
    "mintermins": ["01", "01", "11", "00", "10", "01", "11", "00", "10", "01", "01", "10", "11", "00"],
    "Cluster": ["k0", "k0", "k1", "k1", "k0", "k0", "k1", "k1", "k0", "k0", "k0", "k0", "k1", "k1"]
}
df = pd.DataFrame(data)

# Crear una instancia de FeatureExpander
expander = FeatureExpander(n_variables=1)
resul = expander.fit(df)

values = [1,0]
print("Resultados ",expander.transform(values))
print(resul)