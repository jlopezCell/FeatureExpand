import unittest

from featureexpand.feature_expander import (
    migrate,
    encode,
    generate_variable_map
)

class TestMigrateFunction(unittest.TestCase):


    def test_1v_XOR(self):
        values = [0,1]
        formulaY = [[[1,3]]]
        formulaN = [[[0,2]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)>0)
        self.assertTrue(sum(resultN)==0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

    def test_2v_XORNOT(self):
        values = [1,1]
        formulaY = [[[1,3]]]
        formulaN = [[[0,2]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)==0)
        self.assertTrue(sum(resultN)>0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

    def test_3v_XORNOT(self):
        values = [1,0]
        formulaY = [[[1,3]]]
        formulaN = [[[0,2]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)>0)
        self.assertTrue(sum(resultN)==0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

    def test_4v_XORNOT(self):
        values = [0,0]
        formulaY = [[[1,3]]]
        formulaN = [[[0,2]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)==0)
        self.assertTrue(sum(resultN)>0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))


    def test_5v_XORNOT(self):
        values = [0.4,0.3]
        formulaY = [[[1,3]]]
        formulaN = [[[0,2]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)==0)
        self.assertTrue(sum(resultN)>0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

if __name__ == '__main__':
    unittest.main()