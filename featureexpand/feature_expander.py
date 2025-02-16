import requests
import pandas as pd
import json
from typing import List, Union

def generate_variable_map(variables: List[str]) -> List[str]:
    """
    Generates a variable map by appending an apostrophe to each variable name.

    Args:
        variables (List[str]): List of variable names.

    Returns:
        List[str]: List of modified variable names.

    Raises:
        ValueError: If the input is not a non-empty list.
    """
    if not isinstance(variables, list) or len(variables) == 0:
        raise ValueError("Variables debe ser un array no vacío")
    return [f"{variable}'" for variable in variables]

def number_to_variable(number: int, variable_map: List[str]) -> str:
    """
    Converts a number to a variable name using the variable map.

    Args:
        number (int): Index of the variable in the variable map.
        variable_map (List[str]): List of variable names.

    Returns:
        str: Variable name corresponding to the given index.

    Raises:
        ValueError: If the index is out of range.
    """
    if number < 0 or number >= len(variable_map):
        raise ValueError(f"Índice {number} fuera de rango para el mapa de variables")
    return variable_map[number]

def list_to_logical_expression(lst: Union[List[int], int], variable_map: List[str]) -> str:
    """
    Converts a list of numbers to a logical expression.

    Args:
        lst (Union[List[int], int]): List of numbers or a single number.
        variable_map (List[str]): List of variable names.

    Returns:
        str: Logical expression.

    Raises:
        ValueError: If there is an error in conversion.
    """
    try:
        if not isinstance(lst, list):
            return number_to_variable(lst, variable_map)
        if len(lst) == 0:
            return "TRUE"  # Special case for empty list
        xxy = lst[0]
        negat = "!" if (xxy % 2 == 0) else ""
        expression = negat + "(" + '^'.join(number_to_variable((num - num % 2), variable_map).replace("'", "") for num in lst) + ")"
    except Exception as e:
        raise ValueError(f"Error al convertir la lista a expresión lógica: {e}")
    return expression

def list_to_xor_expression(lst: List[int], variable_map: List[str]) -> str:
    """
    Converts a list of numbers to an XOR expression.

    Args:
        lst (List[int]): List of numbers.
        variable_map (List[str]): List of variable names.

    Returns:
        str: XOR expression.

    Raises:
        TypeError: If the input is not a list.
    """
    if not isinstance(lst, list):
        raise TypeError(f"Se esperaba un array 02, pero se recibió {type(lst).__name__}")
    if len(lst) == 0:
        return ""  # Special case for empty list
    return '&'.join(number_to_variable(num, variable_map) for num in lst)

def transform_function(representation: List[List[int]], variable_map: List[str]) -> str:
    """
    Transforms a logical representation into a logical expression.

    Args:
        representation (List[List[int]]): Logical representation.
        variable_map (List[str]): List of variable names.

    Returns:
        str: Logical expression.

    Raises:
        ValueError: If the input is not a non-empty list or if there is an error in transformation.
    """
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

def encode(numero: float, n: int) -> List[int]:
    """
    Encodes a number into a binary vector of length n.

    Args:
        numero (float): Number to encode.
        n (int): Length of the binary vector.

    Returns:
        List[int]: Binary vector.

    Raises:
        ValueError: If the number is negative.
    """
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

def migrate(values: List[List[float]], 
           nvariables: int, 
           formula: List[List[int]],
           formulaN: List[List[int]] = None
           ) -> List[List[float]]:
    """
    Transforms feature vectors using logical formulas and concatenates with original values.

    Args:
        values (List[List[float]]): Input vectors to transform.
        nvariables (int): Number of variables for encoding.
        formula (List[List[int]]): Logical formula representation.
        formulaN (List[List[int]], optional): Number of formula variables.

    Returns:
        List[List[float]]: List of original vectors concatenated with their transformed versions.
    """
    # Initialize containers for encoded vectors and labels
    vec = []
    labels = []
    
    # Step 1: Encode all input values into binary vectors
    for input_vector in values:
        encoded_vector = []
        for value in input_vector:
            # Extract first element of encoded value (discarding rx string)
            encoded_vector.append(encode(value, nvariables)[0])
        vec.append(encoded_vector)
    
    # Step 2: Process each encoded vector through logical transformation
    result = []
    resultN = []
    logical_expression = ""
    logical_expressionN = ""

    for vector in vec:
        idx = 0
        # Step 2a: Create boolean variables (z0, z1, etc) for logical evaluation
        for j in range(nvariables):        
            for i, value in enumerate(vector):
                globals()[f'z{idx}'] = bool(value == 1)
                # Create positive and negative labels for each variable
                labels.append(f' (not z{idx})')
                labels.append(f'z{idx}')
                idx = idx + 1
        
        # Step 2b: Generate logical expression (only once)
        labels.reverse()
        if not logical_expression:
            logical_expression = transform_function(formula, labels)
            logical_expression = logical_expression.replace("&", " and ").replace("!", " not ")

        if formulaN:
            if not logical_expressionN:
                logical_expressionN = transform_function(formulaN, labels)
                logical_expressionN = logical_expressionN.replace("&", " and ").replace("!", " not ")

        # Step 2c: Evaluate logical expression and convert to numeric
        preset = eval("[" + logical_expression + "]")
        result = [[1.0 if value else 0.0 for value in row] for row in result]
        preset = [1.0 if value else 0.0 for value in preset]
        result.append(preset)

        if formulaN:
            presetN = eval("[" + logical_expressionN + "]")
            resultN = [[1.0 if value else 0.0 for value in row] for row in resultN]
            presetN = [1.0 if value else 0.0 for value in presetN]        
            resultN.append(presetN)

        labels.clear()
    
    # Step 3: Combine original values with transformed results
    if formulaN:
        cb = [original + expanded for original, expanded in zip(values, result)]
        return [original + expanded for original, expanded in zip(cb, resultN)]

    return [original + expanded for original, expanded in zip(values, result)]

class FeatureExpander:
    """
    A class to expand features using logical formulas and encoding.

    Attributes:
        n_variables (int): Number of variables for encoding.
        formula (List[List[int]]): Logical formula for feature expansion.
        token (str): Token for API authentication.
    """

    def __init__(self, token=None, n_variables=None, formula=None):
        """
        Initializes the FeatureExpander class.

        Args:
            token (str, optional): Token for API authentication.
            n_variables (int, optional): Number of variables for encoding.
            formula (List[List[int]], optional): Logical formula for feature expansion.
        """
        self.n_variables = n_variables
        self.formula = formula
        self.token = token

    def fit(self, XInput, y=None, feacture_selection=[], deep=1, response="x1"):
        """
        Fits the model to the input data.

        Args:
            XInput (pd.DataFrame): Input data.
            y (pd.Series, optional): Target labels.
            feacture_selection (List[str], optional): List of features to select.
            precision (int, optional): Precision for encoding.
            response (str, optional): Response variable.

        Raises:
            ValueError: If XInput is not a pandas DataFrame.
        """
        self.n_variables = deep
        X = XInput[feacture_selection]

        if isinstance(X, pd.DataFrame):
            headers = ["mintermins", "Cluster"]
            data = X.values.tolist()
        else:
            raise ValueError("X debe ser un DataFrame de pandas")
        
        X = X.values.tolist()
        y = y.values.tolist()        
        mintermins = []
        for i in range(len(X)):
            rx = ""
            for j in range(len(X[i])):
                rx += "".join(map(str, encode(X[i][j], deep)))
            mintermins.append([rx, y[0][i]])

        json_data = {
            "sheetData": {
                "headers": headers,
                "mintermins": mintermins
            },
            "test": [["Cluster", response]],
            "exclude": ["Cluster"]
        }

        with open("data_from_matrix.json", "w") as json_file:
            json_file.write(json.dumps(json_data, indent=4))

        self.send_data_to_api(json_data)

    def add_features(self, X):
        """
        Expands the features of the input data using the specified formula and number of variables.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            List[List[float]]: Expanded feature vectors.

        Raises:
            ValueError: If n_variables or formula is not specified.
        """
        if isinstance(X, pd.DataFrame):
            headers = X.columns.tolist()
            data = X.values.tolist()
        else:
            data = X
            headers = ["mintermins"]
        if self.n_variables is None or self.formula is None:
            raise ValueError("n_variables y formula deben ser especificados antes de expandir las características.")
        X_expanded = migrate(data, self.n_variables, self.formula, self.formulaN)
        return X_expanded

    def fitOld(self, X, y=None):
        """
        Fits the model to the input data (old version).

        Args:
            X (pd.DataFrame): Input data.
            y (pd.Series, optional): Target labels.

        Returns:
            FeatureExpander: The fitted model.

        Raises:
            ValueError: If X is not a pandas DataFrame.
        """
        if isinstance(X, pd.DataFrame):
            headers = X.columns.tolist()
            data = X.values.tolist()
        else:
            raise ValueError("X debe ser un DataFrame de pandas")

        json_data = {
            "sheetData": {
                "headers": headers,
                "mintermins": data
            },
            "test": [["Cluster", "k0"]],
            "exclude": ["Cluster"]
        }

        self.send_data_to_api(json_data)
        return self

    def send_data_to_api(self, json_data):
        """
        Sends data to a REST API.

        Args:
            json_data (dict): Data to send in JSON format.

        Raises:
            ValueError: If the API response does not contain the expected formulas.
        """
        url = 'https://www.booloptimizer.com/api/simplify'  # Replace with your API URL
        url = 'http://127.0.0.1:5000/api/simplify'  # Replace with your API URL
        bearer_token = self.token
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            result = response.json()
            simplified_expression = result.get("simplified_expression", [])
            
            formulaN = None
            
            for expression in simplified_expression:
                if expression.get("cluster") == "Cumple":
                    formulaYStr = expression.get("result")
                    exp = json.loads(formulaYStr)
                    formula = json.loads(exp.get("result"))
                else:
                    formulaYStr = expression.get("result")
                    exp = json.loads(formulaYStr)
                    formulaN = json.loads(exp.get("result"))
            
            if formula is not None:
                self.formula = formula
            else:
                raise ValueError("No se encontró la fórmula con la etiqueta 'Cumple' en la respuesta de la API.")

            if formulaN is not None:
                self.formulaN = formulaN
            else:
                raise ValueError("No se encontró la fórmula con la etiqueta 'No Cumple' en la respuesta de la API.")

        except requests.exceptions.RequestException as e:
            print(f"Error al enviar datos a la API: {e}")

    def transform(self, X):
        """
        Transforms the input data using the specified formula and number of variables.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            List[List[float]]: Transformed data.

        Raises:
            ValueError: If n_variables or formula is not specified.
        """
        if self.n_variables is None or self.formula is None:
            raise ValueError("n_variables y formula deben ser especificados antes de transformar los datos.")
        X_transformed = migrate(X, self.n_variables, self.formula)
        return X_transformed

    def fit_transform(self, X, y=None):
        """
        Fits the model and transforms the input data in one step.

        Args:
            X (pd.DataFrame): Input data.
            y (pd.Series, optional): Target labels.

        Returns:
            List[List[float]]: Transformed data.
        """
        return self.transform(X)