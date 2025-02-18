import pandas as pd
from featureexpand.feature_expander import FeatureExpander
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Create a sample DataFrame
data = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  1,   1,   0,   1,   0,   1,   0]  # Target variable
}

data2 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   0,   1,   0,   1,   0,   1]  # Target variable
}

data2 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   1,   1,   0,   1,   0,   1]  # Target variable
}

data2 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   1,   1,   0,   1,   0,   1]  # Target variable
}

data2 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            'C': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   1,   1,   0,   1,   1,   1]  # Target variable
}


## TODO agegrar el vector de los valores negativos deben ayudar a mejorar la prediccion
# elf.formula [[[0, 2]]] self.formulaN [[1, 2]]
# X_expanded [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
# self.formula [[[0, 2]]] self.formulaN [[1, 2]]
# Mean Squared Error: 0.44444444444444453 vs Mean Squared Error Expanded: 0.9999999999999996
data2 = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      'Cluster': [  0,   1,   1,   0,   1,   1,   1,   0]  # Empeoro el modelo
}

##yy=pd.Series({'Cluster': ["x1", "x1", "x1", "x1", "x0", "x1", "x1"]})
##yy=pd.Series({'Cluster': ["x1", "x1", "x1", "x1", "x0", "x0", "x0"]})
df = pd.DataFrame(data)
# Split the data into training and testing sets
X = df.drop(columns=['Cluster'])
y = df['Cluster']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

expander = FeatureExpander("Tp6uxDgDHf+meUtDirx0veUq7L59a6M7IsxjRqUJZlc=",enviroment="TEST")
yy=pd.Series({'Cluster': [("x1" if yx == 1 else "x0") for yx in y ]})
expander.fit(X,yy,feacture_selection=["A","B"],deep=1,response="x1")
X_expanded = expander.add_features(X_train)

print("X_expanded",X_expanded)
df = pd.DataFrame(data)
X = df
X = df.drop(columns=['Cluster'])
# Fit a linear regression model
model2 = LinearRegression()
model2.fit(X_expanded, y_train)
# Make predictions
X_test_expanded = expander.add_features(X_test)

y_pred2 = model2.predict(X_test_expanded)
# Evaluate the model
mse2 = mean_squared_error(y_test, y_pred2)

print(f'Mean Squared Error: {mse} vs Mean Squared Error Expanded: {mse2}')
