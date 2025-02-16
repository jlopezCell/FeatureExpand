import unittest
import pandas as pd
from featureexpand.feature_expander import FeatureExpander
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

class TestFeatureExpander(unittest.TestCase):

    def setUp(self):
        # Datos de prueba
        self.data1 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'Cluster': [1, 1, 0, 1, 0, 1, 0]  # Target variable
        }

        self.data2 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'Cluster': [0, 0, 1, 0, 1, 0, 1]  # Target variable
        }

        self.data3 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'Cluster': [0, 1, 1, 0, 1, 0, 1]  # Target variable
        }

        self.data4 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'C': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'Cluster': [0, 1, 1, 0, 1, 1, 1]  # Target variable
        }

        self.data5 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            'Cluster': [0, 1, 1, 0, 1, 1, 1, 0]  # Target variable
        }

        self.feature_selection = ["A", "B"]
        self.precision = 1

    def test_feature_expansion_data1(self):
        self._test_feature_expansion(self.data1)

    def test_feature_expansion_data2(self):
        self._test_feature_expansion(self.data2)

    def test_feature_expansion_data3(self):
        self._test_feature_expansion(self.data3)

    def test_feature_expansion_data4(self):
        self._test_feature_expansion(self.data4)

    def test_feature_expansion_data5(self):
        self._test_feature_expansion(self.data5)

    def _test_feature_expansion(self, data):
        df = pd.DataFrame(data)
        X = df.drop(columns=['Cluster'])
        y = df['Cluster']
        yy = pd.Series({'Cluster': [("x1" if yx == 1 else "x0") for yx in y]})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Modelo sin expansión de características
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Modelo con expansión de características
        expander = FeatureExpander("Tp6uxDgDHf+meUtDirx0veUq7L59a6M7IsxjRqUJZlc=")
        expander.fit(X, yy, self.feature_selection, self.precision, response="x1")
        X_expanded = expander.add_features(X_train)
        model2 = LinearRegression()
        model2.fit(X_expanded, y_train)
        X_test_expanded = expander.add_features(X_test)
        y_pred2 = model2.predict(X_test_expanded)
        mse2 = mean_squared_error(y_test, y_pred2)

        # Verificar que el MSE con expansión de características no sea peor que sin expansión
        self.assertLessEqual(mse2, mse)

    def test_confusion_matrix_data1(self):
        self._test_confusion_matrix(self.data1)

    def test_confusion_matrix_data2(self):
        self._test_confusion_matrix(self.data2)

    def test_confusion_matrix_data3(self):
        self._test_confusion_matrix(self.data3)

    def test_confusion_matrix_data4(self):
        self._test_confusion_matrix(self.data4)

    def test_confusion_matrix_data5(self):
        self._test_confusion_matrix(self.data5)

    def _test_confusion_matrix(self, data):
        df = pd.DataFrame(data)
        y = df['Cluster']
        yy = pd.Series({'Cluster': [("x1" if yx == 1 else "x0") for yx in y]})

        y_true = [(1 if i == "x1" else 0) for i in yy]
        y_pred = [(1 if i == "x1" else 0) for i in yy]

        # Calcular métricas
        cm = confusion_matrix(y_true, y_pred)
        cr = classification_report(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # Verificar que las métricas sean correctas
        self.assertEqual(cm.shape, (2, 2))
        self.assertGreaterEqual(accuracy, 0)
        self.assertGreaterEqual(precision, 0)
        self.assertGreaterEqual(recall, 0)
        self.assertGreaterEqual(f1, 0)

if __name__ == '__main__':
    unittest.main()