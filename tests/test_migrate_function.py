import unittest

from featureexpand.feature_expander import (
    migrate,
    encode,
    generate_variable_map
)

class TestMigrateFunction(unittest.TestCase):
##	{"sheetData":{"headers":["mintermins","Cluster"],"mintermins":[["0111011110","k0"],["1101000000","k1"],["0010110111","k0"],["0101101110","k0"],["0100110000","k1"],["0000101010","k0"],["0101101111","k0"],["1000110000","k0"],["0010111001","k1"],["1000001001","k0"],["1100100100","k0"],["0111101010","k1"],["1001110001","k1"],["0111011100","k1"],["0111010010","k0"],["1111111110","k0"],["1000010101","k1"],["1101100110","k1"],["0000110010","k1"],["0010100100","k0"],["0110100001","k1"],["1111011001","k1"],["1110010011","k1"],["0000001101","k1"],["1111011011","k1"],["1100100110","k1"],["0101100110","k0"],["0011011100","k0"],["0101111000","k1"],["1100011001","k1"],["0000101011","k1"],["1001110000","k0"],["0111101001","k0"],["0110101000","k1"],["1101010011","k1"],["1111000001","k1"],["1001101100","k1"],["1101111101","k1"],["1110110111","k0"],["1011011001","k1"],["0110011000","k0"],["0001001010","k1"],["1001100110","k1"],["1111111110","k1"],["1010010010","k1"],["0000010000","k1"],["1100110001","k0"],["0110101001","k0"],["1101110100","k1"],["0000100101","k0"],["0101010101","k0"],["1111111000","k1"],["1100010100","k0"],["0001101101","k1"],["0010001101","k0"],["0000101011","k1"],["0010100000","k1"],["0010111100","k1"],["0100011000","k0"],["0000011000","k0"],["0110101000","k0"],["1111010000","k0"],["0001000001","k1"],["0000101001","k1"],["1010111001","k1"],["1001100011","k0"],["1001101000","k1"],["0000100010","k1"],["1110010101","k1"],["1101000010","k0"],["1111001110","k1"],["1111000101","k0"],["1110001110","k1"],["1101101001","k1"],["1010000010","k1"],["0010001010","k0"],["1011010011","k0"],["0000010001","k1"],["1101000011","k0"],["1111111101","k1"],["0000101010","k0"],["1011111101","k1"],["0100110101","k0"],["1100001101","k1"],["1001110110","k1"],["0111011100","k1"],["0100111011","k0"],["1100111000","k0"],["0001010001","k1"],["1100100111","k0"],["1010110000","k1"],["0011101001","k0"],["1000100101","k0"],["1101110011","k1"],["1000100011","k1"],["0111001111","k1"],["1000100011","k1"],["0011001001","k0"],["0111000101","k0"],["1100101011","k0"]]},"test":[["Cluster","k0"]],"exclude":["Cluster"]}

    def test_generate_variable_map(self):
        values = [0.1403562556,0.8788954926,0.6819201642,0.8560112024,0.04838814379,0.6785713791,0.7960474374,0.8203090196,0.9930975901,0.3809034321]
        vec = [] 
        nvariables = 1       
        for value in values:
            resultado = encode(value, nvariables)
            vec += resultado
        expected = [0,1,1,1,0,1,1,1,1,0]
        self.assertEqual(vec, expected)

        formulaY = [[3,5,6,9,10,12,14,16,18],[1,2,5,9,10,12,15,16,19],[0,3,5,6,9,11,12,15,19],[2,[1,5],6,9,11,13,15,16,19],[3,4,6,8,11,[1,13],14,17,19],[2,5,7,8,10,12,[0,14],17,19],[3,5,6,8,10,13,14,16,[0,18]],[1,5,7,9,10,[13,3],14,17,19],[0,3,7,9,[10,4],12,14,16,18],[0,3,5,9,11,[13,7],14,16,18],[0,3,5,6,8,13,14,16,[11,19]],[[0,2],[3,5],[5,7],[11,7],8,[10,12],[12,14],17,18],[0,[2,4],[4,6,8],[10,6],[15,17,9],[11,15],12,18],[[1,3],[3,7],4,[7,9],[11,9],12,15,[17,19]],[[1,5],3,[4,8],6,[13,9],11,[12,14,16],19],[[0,2],[2,8],5,7,[12,14,8],11,[13,15,19],17],[[0,4],3,[5,7],[6,8],[15,9],[11,13],[14,16],19],[0,[2,4],[4,8],[11,9],[11,13],14,16,[18,6]],[0,3,6,11,13,14,16,[18,4,8]],[1,[3,7],[5,7,9],[10,4,8],[10,14],13,[14,18],16],[1,[4,6],[6,8],[11,9],[11,13],[12,16],14,[16,18,2]],[1,5,[15,7],9,10,13,[15,17,3],19],[1,7,9,11,12,[15,3],[17,5],19],[1,4,[6,8],11,13,[14,2],17,19],[1,[12,4],7,[3,9],11,[13,15],[15,17],19],[1,[13,3,5],7,8,11,[13,17],15,19],[[2,8],5,7,[0,10,8],[1,11,13],[13,15],[14,16],19],[2,5,[1,7,9],[0,10,8],[11,13],15,17,19],[3,[4,8],[0,6],[11,9],[11,13],14,16,18],[[1,3],[13,3],7,8,10,[13,15],[15,17],[16,18]],[[1,5],[13,5],6,8,11,[12,16],14,[16,18]]]
        formulaN = [[0,2,5,7,8,11,12,17,18],[[1,3],5,7,8,11,13,15,17,19],[2,4,7,8,11,[1,13],14,17,18],[3,4,7,8,11,13,[0,14],17,18],[0,[2,4],7,8,11,12,15,16,18],[1,4,[3,7],9,11,12,14,17,18],[0,5,7,[2,8],11,12,15,17,18],[1,5,6,8,11,[12,2],15,17,19],[0,4,7,9,10,13,[14,2],16,18],[0,3,5,[7,9],11,13,15,17,19],[1,3,4,8,[11,7],13,15,16,18],[0,3,5,9,11,13,15,17,[19,7]],[1,2,5,[6,8],[11,9],[11,13],[13,15],[15,17],18],[[0,4],2,[5,7],9,11,13,14,17,18],[1,[3,5],[5,7],[6,8],[10,8],[11,13],[12,14],[15,17],[17,19]],[[1,3,9],[3,7,9],5,[14,6],[16,8],11,13,[17,19]],[[1,3],[2,4],[5,7],[11,7],[10,14],13,[14,16],[17,19,9]],[[1,3],[2,4],[5,7],[13,7,9],11,[13,17,9],15,[17,19]],[0,[2,6],[10,6],9,[10,12],[12,14],[15,17],[17,19,5]],[0,[10,2],7,9,[11,13],[14,4],16,18],[[0,12],2,[4,6],8,11,[12,14],[14,16],[16,18]],[[2,6],4,[11,7],8,[10,12],14,16,[1,19]],[3,[4,6],[13,7],8,11,[13,17],14,[1,19]],[2,4,[14,6],9,10,13,[14,18],[1,17]],[[1,7],2,4,[11,7],[11,19],13,15,16],[[1,3],[13,3],4,6,11,[13,17],15,[16,18]],[0,[15,3],4,6,11,13,[15,17],[16,18]]]



    def test_typical_case(self):
        values = [0.5, 0.25]
        nvariables = 2
        formula = [[0, 1], [1]]
        result = migrate(values, nvariables, formula)
        print(result)
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

    def test_1v_nvariables(self):
        values = [0.1403562556,0.8788954926,0.6819201642,0.8560112024,0.04838814379,0.6785713791,0.7960474374,0.8203090196,0.9930975901,0.3809034321]
        formulaY = [[3,5,6,9,10,12,14,16,18],[1,2,5,9,10,12,15,16,19],[0,3,5,6,9,11,12,15,19],[2,[1,5],6,9,11,13,15,16,19],[3,4,6,8,11,[1,13],14,17,19],[2,5,7,8,10,12,[0,14],17,19],[3,5,6,8,10,13,14,16,[0,18]],[1,5,7,9,10,[13,3],14,17,19],[0,3,7,9,[10,4],12,14,16,18],[0,3,5,9,11,[13,7],14,16,18],[0,3,5,6,8,13,14,16,[11,19]],[[0,2],[3,5],[5,7],[11,7],8,[10,12],[12,14],17,18],[0,[2,4],[4,6,8],[10,6],[15,17,9],[11,15],12,18],[[1,3],[3,7],4,[7,9],[11,9],12,15,[17,19]],[[1,5],3,[4,8],6,[13,9],11,[12,14,16],19],[[0,2],[2,8],5,7,[12,14,8],11,[13,15,19],17],[[0,4],3,[5,7],[6,8],[15,9],[11,13],[14,16],19],[0,[2,4],[4,8],[11,9],[11,13],14,16,[18,6]],[0,3,6,11,13,14,16,[18,4,8]],[1,[3,7],[5,7,9],[10,4,8],[10,14],13,[14,18],16],[1,[4,6],[6,8],[11,9],[11,13],[12,16],14,[16,18,2]],[1,5,[15,7],9,10,13,[15,17,3],19],[1,7,9,11,12,[15,3],[17,5],19],[1,4,[6,8],11,13,[14,2],17,19],[1,[12,4],7,[3,9],11,[13,15],[15,17],19],[1,[13,3,5],7,8,11,[13,17],15,19],[[2,8],5,7,[0,10,8],[1,11,13],[13,15],[14,16],19],[2,5,[1,7,9],[0,10,8],[11,13],15,17,19],[3,[4,8],[0,6],[11,9],[11,13],14,16,18],[[1,3],[13,3],7,8,10,[13,15],[15,17],[16,18]],[[1,5],[13,5],6,8,11,[12,16],14,[16,18]]]
        formulaN = [[0,2,5,7,8,11,12,17,18],[[1,3],5,7,8,11,13,15,17,19],[2,4,7,8,11,[1,13],14,17,18],[3,4,7,8,11,13,[0,14],17,18],[0,[2,4],7,8,11,12,15,16,18],[1,4,[3,7],9,11,12,14,17,18],[0,5,7,[2,8],11,12,15,17,18],[1,5,6,8,11,[12,2],15,17,19],[0,4,7,9,10,13,[14,2],16,18],[0,3,5,[7,9],11,13,15,17,19],[1,3,4,8,[11,7],13,15,16,18],[0,3,5,9,11,13,15,17,[19,7]],[1,2,5,[6,8],[11,9],[11,13],[13,15],[15,17],18],[[0,4],2,[5,7],9,11,13,14,17,18],[1,[3,5],[5,7],[6,8],[10,8],[11,13],[12,14],[15,17],[17,19]],[[1,3,9],[3,7,9],5,[14,6],[16,8],11,13,[17,19]],[[1,3],[2,4],[5,7],[11,7],[10,14],13,[14,16],[17,19,9]],[[1,3],[2,4],[5,7],[13,7,9],11,[13,17,9],15,[17,19]],[0,[2,6],[10,6],9,[10,12],[12,14],[15,17],[17,19,5]],[0,[10,2],7,9,[11,13],[14,4],16,18],[[0,12],2,[4,6],8,11,[12,14],[14,16],[16,18]],[[2,6],4,[11,7],8,[10,12],14,16,[1,19]],[3,[4,6],[13,7],8,11,[13,17],14,[1,19]],[2,4,[14,6],9,10,13,[14,18],[1,17]],[[1,7],2,4,[11,7],[11,19],13,15,16],[[1,3],[13,3],4,6,11,[13,17],15,[16,18]],[0,[15,3],4,6,11,13,[15,17],[16,18]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)>0)
        self.assertTrue(sum(resultN)==0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))
        
    def test_1v_nvariables_1(self):
        values = [0.5219193757,0.6836017936,0.3468377486,0.9339642619,0.457964091,0.4725818157,0.3991877202,0.4391629039,0.3999209042,0.2181663905]
        formulaY = [[3,5,6,9,10,12,14,16,18],[1,2,5,9,10,12,15,16,19],[0,3,5,6,9,11,12,15,19],[2,[1,5],6,9,11,13,15,16,19],[3,4,6,8,11,[1,13],14,17,19],[2,5,7,8,10,12,[0,14],17,19],[3,5,6,8,10,13,14,16,[0,18]],[1,5,7,9,10,[13,3],14,17,19],[0,3,7,9,[10,4],12,14,16,18],[0,3,5,9,11,[13,7],14,16,18],[0,3,5,6,8,13,14,16,[11,19]],[[0,2],[3,5],[5,7],[11,7],8,[10,12],[12,14],17,18],[0,[2,4],[4,6,8],[10,6],[15,17,9],[11,15],12,18],[[1,3],[3,7],4,[7,9],[11,9],12,15,[17,19]],[[1,5],3,[4,8],6,[13,9],11,[12,14,16],19],[[0,2],[2,8],5,7,[12,14,8],11,[13,15,19],17],[[0,4],3,[5,7],[6,8],[15,9],[11,13],[14,16],19],[0,[2,4],[4,8],[11,9],[11,13],14,16,[18,6]],[0,3,6,11,13,14,16,[18,4,8]],[1,[3,7],[5,7,9],[10,4,8],[10,14],13,[14,18],16],[1,[4,6],[6,8],[11,9],[11,13],[12,16],14,[16,18,2]],[1,5,[15,7],9,10,13,[15,17,3],19],[1,7,9,11,12,[15,3],[17,5],19],[1,4,[6,8],11,13,[14,2],17,19],[1,[12,4],7,[3,9],11,[13,15],[15,17],19],[1,[13,3,5],7,8,11,[13,17],15,19],[[2,8],5,7,[0,10,8],[1,11,13],[13,15],[14,16],19],[2,5,[1,7,9],[0,10,8],[11,13],15,17,19],[3,[4,8],[0,6],[11,9],[11,13],14,16,18],[[1,3],[13,3],7,8,10,[13,15],[15,17],[16,18]],[[1,5],[13,5],6,8,11,[12,16],14,[16,18]]]
        formulaN = [[0,2,5,7,8,11,12,17,18],[[1,3],5,7,8,11,13,15,17,19],[2,4,7,8,11,[1,13],14,17,18],[3,4,7,8,11,13,[0,14],17,18],[0,[2,4],7,8,11,12,15,16,18],[1,4,[3,7],9,11,12,14,17,18],[0,5,7,[2,8],11,12,15,17,18],[1,5,6,8,11,[12,2],15,17,19],[0,4,7,9,10,13,[14,2],16,18],[0,3,5,[7,9],11,13,15,17,19],[1,3,4,8,[11,7],13,15,16,18],[0,3,5,9,11,13,15,17,[19,7]],[1,2,5,[6,8],[11,9],[11,13],[13,15],[15,17],18],[[0,4],2,[5,7],9,11,13,14,17,18],[1,[3,5],[5,7],[6,8],[10,8],[11,13],[12,14],[15,17],[17,19]],[[1,3,9],[3,7,9],5,[14,6],[16,8],11,13,[17,19]],[[1,3],[2,4],[5,7],[11,7],[10,14],13,[14,16],[17,19,9]],[[1,3],[2,4],[5,7],[13,7,9],11,[13,17,9],15,[17,19]],[0,[2,6],[10,6],9,[10,12],[12,14],[15,17],[17,19,5]],[0,[10,2],7,9,[11,13],[14,4],16,18],[[0,12],2,[4,6],8,11,[12,14],[14,16],[16,18]],[[2,6],4,[11,7],8,[10,12],14,16,[1,19]],[3,[4,6],[13,7],8,11,[13,17],14,[1,19]],[2,4,[14,6],9,10,13,[14,18],[1,17]],[[1,7],2,4,[11,7],[11,19],13,15,16],[[1,3],[13,3],4,6,11,[13,17],15,[16,18]],[0,[15,3],4,6,11,13,[15,17],[16,18]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)==0)
        self.assertTrue(sum(resultN)>0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

    def test_1v_nvariables_2(self):
        values = [0.3282859553,0.6931947352,0.4114704204,0.8042826988,0.764317795,0.2997029924,0.5446668477,0.8219970857,0.9161249237,0.296546135]
        formulaY = [[3,5,6,9,10,12,14,16,18],[1,2,5,9,10,12,15,16,19],[0,3,5,6,9,11,12,15,19],[2,[1,5],6,9,11,13,15,16,19],[3,4,6,8,11,[1,13],14,17,19],[2,5,7,8,10,12,[0,14],17,19],[3,5,6,8,10,13,14,16,[0,18]],[1,5,7,9,10,[13,3],14,17,19],[0,3,7,9,[10,4],12,14,16,18],[0,3,5,9,11,[13,7],14,16,18],[0,3,5,6,8,13,14,16,[11,19]],[[0,2],[3,5],[5,7],[11,7],8,[10,12],[12,14],17,18],[0,[2,4],[4,6,8],[10,6],[15,17,9],[11,15],12,18],[[1,3],[3,7],4,[7,9],[11,9],12,15,[17,19]],[[1,5],3,[4,8],6,[13,9],11,[12,14,16],19],[[0,2],[2,8],5,7,[12,14,8],11,[13,15,19],17],[[0,4],3,[5,7],[6,8],[15,9],[11,13],[14,16],19],[0,[2,4],[4,8],[11,9],[11,13],14,16,[18,6]],[0,3,6,11,13,14,16,[18,4,8]],[1,[3,7],[5,7,9],[10,4,8],[10,14],13,[14,18],16],[1,[4,6],[6,8],[11,9],[11,13],[12,16],14,[16,18,2]],[1,5,[15,7],9,10,13,[15,17,3],19],[1,7,9,11,12,[15,3],[17,5],19],[1,4,[6,8],11,13,[14,2],17,19],[1,[12,4],7,[3,9],11,[13,15],[15,17],19],[1,[13,3,5],7,8,11,[13,17],15,19],[[2,8],5,7,[0,10,8],[1,11,13],[13,15],[14,16],19],[2,5,[1,7,9],[0,10,8],[11,13],15,17,19],[3,[4,8],[0,6],[11,9],[11,13],14,16,18],[[1,3],[13,3],7,8,10,[13,15],[15,17],[16,18]],[[1,5],[13,5],6,8,11,[12,16],14,[16,18]]]
        formulaN = [[0,2,5,7,8,11,12,17,18],[[1,3],5,7,8,11,13,15,17,19],[2,4,7,8,11,[1,13],14,17,18],[3,4,7,8,11,13,[0,14],17,18],[0,[2,4],7,8,11,12,15,16,18],[1,4,[3,7],9,11,12,14,17,18],[0,5,7,[2,8],11,12,15,17,18],[1,5,6,8,11,[12,2],15,17,19],[0,4,7,9,10,13,[14,2],16,18],[0,3,5,[7,9],11,13,15,17,19],[1,3,4,8,[11,7],13,15,16,18],[0,3,5,9,11,13,15,17,[19,7]],[1,2,5,[6,8],[11,9],[11,13],[13,15],[15,17],18],[[0,4],2,[5,7],9,11,13,14,17,18],[1,[3,5],[5,7],[6,8],[10,8],[11,13],[12,14],[15,17],[17,19]],[[1,3,9],[3,7,9],5,[14,6],[16,8],11,13,[17,19]],[[1,3],[2,4],[5,7],[11,7],[10,14],13,[14,16],[17,19,9]],[[1,3],[2,4],[5,7],[13,7,9],11,[13,17,9],15,[17,19]],[0,[2,6],[10,6],9,[10,12],[12,14],[15,17],[17,19,5]],[0,[10,2],7,9,[11,13],[14,4],16,18],[[0,12],2,[4,6],8,11,[12,14],[14,16],[16,18]],[[2,6],4,[11,7],8,[10,12],14,16,[1,19]],[3,[4,6],[13,7],8,11,[13,17],14,[1,19]],[2,4,[14,6],9,10,13,[14,18],[1,17]],[[1,7],2,4,[11,7],[11,19],13,15,16],[[1,3],[13,3],4,6,11,[13,17],15,[16,18]],[0,[15,3],4,6,11,13,[15,17],[16,18]]]
        nvariables = 1
        resultY = migrate(values, nvariables, formulaY)
        resultN = migrate(values, nvariables, formulaN)
        print(resultY,resultN)
        # valida que resultY sumado sea mayor que 0
        self.assertTrue(sum(resultY)>0)
        self.assertTrue(sum(resultN)==0)
        self.assertIsInstance(resultY, list)
        self.assertTrue(all(x in [0, 1] for x in resultY))

"""     def test_out_of_range_indices(self):
        values = [0.5, 0.25]
        nvariables = 2
        formula = [[0, 2], [1]]
        with self.assertRaises(ValueError):
            migrate(values, nvariables, formula) """

if __name__ == '__main__':
    unittest.main()