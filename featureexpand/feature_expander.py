import unittest
import requests
import pandas as pd
import json

def generate_variable_map(variables):
    if not isinstance(variables, list) or len(variables) == 0:
        raise ValueError("Variables debe ser un array no vacío")
    return [f"{variable}'" for variable in variables]

def number_to_variable(number, variable_map):
    if number < 0 or number >= len(variable_map):
        raise ValueError(f"Índice {number} fuera de rango para el mapa de variables")
    return variable_map[number]

def list_to_logical_expression(lst, variable_map):
    if not isinstance(lst, list):
        return number_to_variable(lst, variable_map)
    if len(lst) == 0:
        return "TRUE"  # Caso especial para lista vacía
    negat = "!" if lst[0] % 2 == 0 else ""
    expression = negat + "(" + '^'.join(number_to_variable(num, variable_map).replace("'", "") for num in lst) + ")" 
    return expression

def list_to_xor_expression(lst, variable_map):
    if not isinstance(lst, list):
        raise TypeError(f"Se esperaba un array 02, pero se recibió {type(lst).__name__}")
    if len(lst) == 0:
        return ""  # Caso especial para lista vacía
    return '&'.join(number_to_variable(num, variable_map) for num in lst)

def transform_function(representation, variable_map):
    if not isinstance(representation, list) or len(representation) == 0:
        raise ValueError("La representación debe ser un array no vacío")
    try:
        expressions = []
        for term in representation:
            if not isinstance(term, list) or len(term) < 1:
                raise ValueError("Cada término debe ser un array no vacío")
            output = ""
            for element in term:
                and_part = element
                and_expression = list_to_logical_expression(and_part, variable_map)
                output += ("" if output == "" else "&") + and_expression
            expressions.append(output)
        return ','.join(expressions)
    except Exception as error:
        raise ValueError(f"Error al transformar la función: {error}")

def encode(numero, n):
    if numero < 0:
        raise ValueError("El valor no puede ser negativo")

    resto = numero
    digitos = []
    limite = 0.5
    for i in range(n):
        if resto > limite:
            resto = resto - limite
            digitos.append(1)
        else:
            digitos.append(0)
        limite = limite * 0.5
    return digitos

def migrate(values, nvariables, formula):
    # Encode all values
    vec = []
    labels = []

    X = values    
    vec = []
    for i in range(len(X)):
        vecs = []
        rx = ""
        for j in range(len(X[i])):
            rx += "".join(map(str, encode(X[i][j],nvariables)))
            vecs.append(encode(X[i][j],nvariables)[0])
        vec.append(vecs)
    result = []    
    for vector in vec:    
        # Create labels and set global variables
        for i, valor in enumerate(vector):
            globals()[f'z{i}'] = True if valor == 1 else False
            labels.append(f' (not z{i})')
            labels.append(f'z{i}')
    
        labels.reverse()
        # Transform the formula into a logical expression
        logical_expression = transform_function(formula, labels)
        logical_expression = logical_expression.replace("&", " and ").replace("!", " not ")
        print(z0,z1,logical_expression)
        # Evaluate the logical expression
        result.append(eval("[" + logical_expression + "]"))
    
    # Convert the result to a list of binary values
    return result


class FeatureExpander:
    def __init__(self, n_variables=None, formula=None):
        """
        Constructor de la clase FeatureExpander.

        Parámetros:
        -----------
        n_variables : int, opcional
            Número de variables a utilizar en la expansión de características.
        formula : list, opcional
            Fórmula lógica que define cómo se expandirán las características.
        """
        self.n_variables = n_variables
        self.formula = formula

    def fit(self, X, y=None, precision=1, response="x1"):
        self.n_variables = precision
        print( X, y, precision)
        # Simular la obtención de datos de una hoja de cálculo
        # Supongamos que X es un DataFrame de pandas con los datos de la hoja de cálculo
        if isinstance(X, pd.DataFrame):
            ## La variable  y que es una serie, extrae el nombre de la columna y lo añade a la lista de headers
            headers =  ["mintermins","Cluster"]
            data = X.values.tolist()
        else:
            raise ValueError("X debe ser un DataFrame de pandas")
        
        ## X es un DataFrame de pandas con los datos de la hoja de cálculo, convertelo en una lista de listas
        
        X = X.values.tolist()
        ## X = X transpose
        ##X = list(map(list, zip(*X)))
        y = y.values.tolist()        
        mintermins = []
        for i in range(len(X)):
            rx = ""
            for j in range(len(X[i])):
                rx += "".join(map(str, encode(X[i][j],precision)))
            mintermins.append([rx,y[0][i]])

        # Generar JSON de los datos
        json_data = {
            "sheetData": {
                "headers": headers,
                "mintermins": mintermins  # Aquí podrías procesar los datos si es necesario
            },
            "test": [["Cluster", response]],
            "exclude": ["Cluster"]
        }

        # Guardar el JSON en un archivo (opcional)
        with open("data_from_matrix.json", "w") as json_file:
            json_file.write(json.dumps(json_data, indent=4))
        print("Envio ", json.dumps(json_data, indent=4))

        # Enviar JSON por POST a un servicio REST API
        self.send_data_to_api(json_data)


    def add_features(self, X):
        """
        Expande las características de los datos de entrada utilizando la fórmula y el número de variables especificados.

        Parámetros:
        -----------
        X : array-like
            Datos de entrada.

        Retorna:
        --------
        X_expanded : array-like
            Datos de entrada con características expandidas.
        """
        print("PASO 1", X)

        #quiero verificar que X sea un DataFrame de pandas
        if isinstance(X, pd.DataFrame):
            headers = X.columns.tolist()
            data = X.values.tolist()
        else:
            data = X
            headers = ["mintermins"]
        if self.n_variables is None or self.formula is None:
            raise ValueError("n_variables y formula deben ser especificados antes de expandir las características.")
        # Aquí podrías llamar a la función `migrate` o cualquier otra lógica de expansión.
        X_expanded = migrate(data, self.n_variables, self.formula)
        print(X_expanded)
        return X_expanded


    def fitOld(self, X, y=None):
        """
        Ajusta el modelo a los datos de entrada.

        Parámetros:
        -----------
        X : array-like
            Datos de entrada.
        y : array-like, opcional
            Etiquetas de los datos de entrada. No se utiliza en este caso, pero se incluye para mantener la compatibilidad con Scikit-learn.

        Retorna:
        --------
        self : FeatureExpander
            Instancia del modelo ajustado.
        """

        

        # Simular la obtención de datos de una hoja de cálculo
        # Supongamos que X es un DataFrame de pandas con los datos de la hoja de cálculo
        if isinstance(X, pd.DataFrame):
            headers = X.columns.tolist()
            data = X.values.tolist()
        else:
            raise ValueError("X debe ser un DataFrame de pandas")

        print("Envio ", headers,"XXXXX", data)
        # Generar JSON de los datos
        json_data = {
            "sheetData": {
                "headers": headers,
                "mintermins": data  # Aquí podrías procesar los datos si es necesario
            },
            "test": [["Cluster", "k0"]],
            "exclude": ["Cluster"]
        }

        # Enviar JSON por POST a un servicio REST API
        self.send_data_to_api(json_data)

        return self



    def send_data_to_api(self, json_data):
        """
        Envía los datos a una API REST.
        Parámetros:
        -----------
        json_data : dict
            Datos en formato JSON para enviar a la API.
        """
        url = 'https://www.booloptimizer.com/api/simplify'  # Reemplaza con la URL de tu API
        url = 'http://127.0.0.1:5000/api/simplify'  # Reemplaza con la URL de tu API
        bearer_token = "39c201cd-6b66-4458-aa38-504e25acdfe1"
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, headers=headers, json=json_data)
            response.raise_for_status()  # Lanza una excepción si la respuesta no es 200
           ## print("Respuesta de la API:", response.json())
            result = response.json()
            # Extract the simplified expression from the API response
            simplified_expression = result.get("simplified_expression", [])
            
            # Initialize variables to store the formulas
            formulaN = None
            
            # Iterate over the simplified expression to find the formulas
            for expression in simplified_expression:
                if expression.get("cluster") == "Cumple":
                    formulaYStr = expression.get("result")
                    exp = json.loads(formulaYStr)
                    formula = json.loads(exp.get("result"))
                else:
                    formulaYStr = expression.get("result")
                    exp = json.loads(formulaYStr)
                    formulaN = json.loads(exp.get("result"))
            
            # Assign the formulaY to self.formula
            if formula is not None:
                self.formula = formula
            else:
                raise ValueError("No se encontró la fórmula con la etiqueta 'Cumple' en la respuesta de la API.")
            # Assign the formulaY to self.formula

            if formulaN is not None:
                self.formulaN = formulaN
            else:
                raise ValueError("No se encontró la fórmula con la etiqueta 'No Cumple' en la respuesta de la API.")

        except requests.exceptions.RequestException as e:
            print(f"Error al enviar datos a la API: {e}")


    def transform(self, X):
        """
        Transforma los datos de entrada utilizando la fórmula y el número de variables especificados.

        Parámetros:
        -----------
        X : array-like
            Datos de entrada.

        Retorna:
        --------
        X_transformed : array-like
            Datos transformados.
        """
        if self.n_variables is None or self.formula is None:
            raise ValueError("n_variables y formula deben ser especificados antes de transformar los datos.")
        # Aquí podrías llamar a la función `migrate` o cualquier otra lógica de transformación.
        X_transformed = migrate(X, self.n_variables, self.formula)
##        N_transformed = migrate(X, self.n_variables, self.formulaN)
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Ajusta el modelo y transforma los datos de entrada en un solo paso.

        Parámetros:
        -----------
        X : array-like
            Datos de entrada.
        y : array-like, opcional
            Etiquetas de los datos de entrada.

        Retorna:
        --------
        X_transformed : array-like
            Datos transformados.
        """
        ##self.fit(X, y)
        return self.transform(X)