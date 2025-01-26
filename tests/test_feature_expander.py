import unittest
from featureexpand.feature_expander import (
    generate_variable_map,
    number_to_variable,
    list_to_logical_expression,
    list_to_xor_expression,
    transform_function,
    encode
)

class TestFeatureExpander(unittest.TestCase):

    def test_generate_variable_map(self):
        # Caso normal
        variables = ['a', 'b', 'c']
        expected = ["a'", "b'", "c'"]
        self.assertEqual(generate_variable_map(variables), expected)

        # Caso con lista vacía
        with self.assertRaises(ValueError):
            generate_variable_map([])

        # Caso con tipo incorrecto
        with self.assertRaises(ValueError):
            generate_variable_map("not a list")

    def test_number_to_variable(self):
        variable_map = ["a'", "b'", "c'"]
        
        # Caso normal
        self.assertEqual(number_to_variable(0, variable_map), "a'")
        self.assertEqual(number_to_variable(1, variable_map), "b'")
        self.assertEqual(number_to_variable(2, variable_map), "c'")

        # Caso con índice fuera de rango
        with self.assertRaises(ValueError):
            number_to_variable(3, variable_map)

        # Caso con índice negativo
        with self.assertRaises(ValueError):
            number_to_variable(-1, variable_map)

    def test_list_to_logical_expression(self):
        variable_map = ["a'", "b'", "c'"]
        
        # Caso con lista vacía
        self.assertEqual(list_to_logical_expression([], variable_map), "TRUE")

        # Caso con lista de un solo elemento
        self.assertEqual(list_to_logical_expression([0], variable_map), "!(a)")
        self.assertEqual(list_to_logical_expression([1], variable_map), "(b)")

        # Caso con lista de múltiples elementos
        self.assertEqual(list_to_logical_expression([0, 1], variable_map), "!(a^b)")
        self.assertEqual(list_to_logical_expression([0, 1, 2], variable_map), "!(a^b^c)")

        # Caso con número negativo (no debería ocurrir, pero se prueba)
        with self.assertRaises(ValueError):
            list_to_logical_expression([-1], variable_map)

    def test_list_to_xor_expression(self):
        variable_map = ["a'", "b'", "c'"]
        
        # Caso con lista vacía
        self.assertEqual(list_to_xor_expression([], variable_map), "")

        # Caso con lista de un solo elemento
        self.assertEqual(list_to_xor_expression([0], variable_map), "a'")
        self.assertEqual(list_to_xor_expression([1], variable_map), "b'")

        # Caso con lista de múltiples elementos
        self.assertEqual(list_to_xor_expression([0, 1], variable_map), "a'&b'")
        self.assertEqual(list_to_xor_expression([0, 1, 2], variable_map), "a'&b'&c'")

        # Caso con tipo incorrecto
        with self.assertRaises(TypeError):
            list_to_xor_expression("not a list", variable_map)


    def test_encode_with_0_84_and_n_5(self):
        # Prueba la función con numero = 0.84 y n = 5
        resultado = encode(0.84, 5)
        print(resultado)  # Esto imprimirá la lista de dígitos resultante
        # Verifica que el resultado sea el esperado
        self.assertEqual(resultado, [1, 1, 0, 1, 0])  # Ajusta este valor según la lógica de encode


    def test_transform_function(self):
        variable_map = ["a'", "b'", "c'"]
        
        # Caso normal
        representation = [[[0, 1], [2]], [[1, 2]]]
        expected = "!(a^b)&!(c),(b^c)"
        self.assertEqual(transform_function(representation, variable_map), expected)

        # Caso con lista vacía
        with self.assertRaises(ValueError):
            transform_function([], variable_map)

        # Caso con tipo incorrecto
        with self.assertRaises(ValueError):
            transform_function("not a list", variable_map)

        # Caso con término vacío
        with self.assertRaises(ValueError):
            transform_function([[]], variable_map)

    def test_encode(self):
        # Caso normal
        self.assertEqual(encode(0.75, 3), [1, 0, 1])
        self.assertEqual(encode(0.5, 3), [0, 1, 1])
        self.assertEqual(encode(0.25, 3), [0, 0, 1])
        self.assertEqual(encode(0.125, 3), [0, 0, 0])

        # Caso con número negativo (no debería ocurrir, pero se prueba)
        with self.assertRaises(ValueError):
            encode(-0.5, 3)

        # Caso con n=0
        self.assertEqual(encode(0.5, 0), [])

if __name__ == '__main__':
    unittest.main()
