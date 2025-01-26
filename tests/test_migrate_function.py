import unittest

from featureexpand.feature_expander import (
    migrate,
    generate_variable_map
)

class TestMigrateFunction(unittest.TestCase):
    
    def test_typical_case(self):
        values = [0.5, 0.25]
        nvariables = 2
        formula = [[0, 1], [1]]
        result = migrate(values, nvariables, formula)
        self.assertIsInstance(result, list)
        self.assertTrue(all(x in [0, 1] for x in result))
    
    def test_negative_values(self):
        values = [-0.5, 0.25]
        nvariables = 2
        formula = [[0, 1], [1]]
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula)
    
    def test_zero_nvariables(self):
        values = [0.5, 0.25]
        nvariables = 0
        formula = [[0, 1], [1]]
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula)
    
    def test_empty_formula(self):
        values = [0.5, 0.25]
        nvariables = 2
        formula = []
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula)
    
    def test_empty_values(self):
        values = []
        nvariables = 2
        formula = [[0, 1], [1]]
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula)
    
    def test_invalid_formula(self):
        values = [0.5, 0.25]
        nvariables = 2
        formula = [0, 1]
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula)
    
    def test_large_nvariables(self):
        values = [0.5, 0.25]
        nvariables = 10
        formula = [[0, 1], [1]]
        result = migrate(values, nvariables, formula)
        self.assertIsInstance(result, list)
        self.assertTrue(all(x in [0, 1] for x in result))
    
    def test_out_of_range_indices(self):
        values = [0.5, 0.25]
        nvariables = 2
        formula = [[0, 2], [1]]
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula)

if __name__ == '__main__':
    unittest.main()