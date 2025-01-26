import unittest
from featureexpand.feature_expander import FeatureExpander, migrate, encode, generate_variable_map

class TestMigrateFunction(unittest.TestCase):
    def test_1v_XOR(self):
        values = [0, 1]
        formulaY = [[[1, 3]]]
        formulaN = [[[0, 2]]]
        nvariables = 1

        # Crear una instancia de FeatureExpander
        expanderY = FeatureExpander(n_variables=nvariables, formula=formulaY)
        expanderN = FeatureExpander(n_variables=nvariables, formula=formulaN)

        # Transformar los datos
        resultY = expanderY.fit_transform(values)
        resultN = expanderN.fit_transform(values)

        print("Imprimir",resultY, resultN)

        # Validar los resultados
        self.assertTrue(sum(resultY) > 0)
        self.assertTrue(sum(resultN) == 0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

if __name__ == '__main__':
    unittest.main()