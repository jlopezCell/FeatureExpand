import unittest

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
    
    for value in values:
        resultado = encode(value, nvariables)
        vec += resultado
    
    print(vec)

    # Create labels and set global variables
    for i, valor in enumerate(vec):
        globals()[f'z{i}'] = True if valor == 1 else False
        labels.append(f' (not z{i})')
        labels.append(f'z{i}')
    
    labels.reverse()
    # Transform the formula into a logical expression
    logical_expression = transform_function(formula, labels)
    logical_expression = logical_expression.replace("&", " and ").replace("!", " not ")
    
    # Evaluate the logical expression
    result = eval("[" + logical_expression + "]")
    
    # Convert the result to a list of binary values
    return [1 if x else 0 for x in result]


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

    def fit(self, X, y=None):
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
        # Aquí podrías realizar cualquier ajuste necesario basado en los datos de entrada.
        # Por ejemplo, calcular estadísticas, ajustar parámetros, etc.
        return self

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
        self.fit(X, y)
        return self.transform(X)