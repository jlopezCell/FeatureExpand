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

data = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   0,   1,   0,   1,   0,   1]  # Target variable
}

datas = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   1,   1,   0,   1,   0,   1]  # Target variable
}

datas = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   1,   1,   0,   1,   0,   1]  # Target variable
}

data = {
            'A': [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            'B': [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
      'Cluster': [  0,   1,   1,   0,   1,   1,   1]  # Target variable
}


## TODO agegrar el vector de los valores negativos deben ayudar a mejorar la prediccion
# elf.formula [[[0, 2]]] self.formulaN [[1, 2]]
# X_expanded [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
# self.formula [[[0, 2]]] self.formulaN [[1, 2]]
# Mean Squared Error: 0.44444444444444453 vs Mean Squared Error Expanded: 0.9999999999999996
datas = {
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
yy=pd.Series({'Cluster': [("x1" if yx == 1 else "x0") for yx in y ]})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
precision = 1
# Initialize the FeatureExpander
expander = FeatureExpander()
# Add new features
expander.fit(X,yy,precision,response="x1")
values = [[1,0]]
##print("Resultados ",expander.transform(values))
df = pd.DataFrame(data)
X = df
X = df.drop(columns=['Cluster'])
X_expanded = expander.add_features(X_train)
print("X_expanded",X_expanded)
# Fit a linear regression model
model2 = LinearRegression()
model2.fit(X_expanded, y_train)
# Make predictions
X_test_expanded = expander.add_features(X_test)

y_pred2 = model2.predict(X_test_expanded)
# Evaluate the model
mse2 = mean_squared_error(y_test, y_pred2)


print(f'Mean Squared Error: {mse} vs Mean Squared Error Expanded: {mse2}')


exit(0)
## Quiero hacer un matrix de confusion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

## itera sobre yy y convertir a 0 y 1
y_true= [(1 if i=="x1" else 0) for i in yy]
y_pred= [(1 if i=="x1" else 0) for i in yy]
##y_pred2 = [(1 if sum(i)>0 else 0) for i in [( 1 if y else 0) for y in X_expanded]]
confusion_matrix(y_true, y_pred)
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))
print("Classification Report")
print(classification_report(y_true, y_pred))
print("Accuracy Score")
print(accuracy_score(y_true, y_pred))
print("Precision Score")
print(precision_score(y_true, y_pred, average='macro'))
print("Recall Score")
print(recall_score(y_true, y_pred, average='macro'))
print("F1 Score")
print(f1_score(y_true, y_pred, average='macro'))

exit(0)
# Initialize the FeatureExpander
expander = FeatureExpander()
# Add new features
X_expanded = expander.add_features(X,y,precision)
X_train_expanded, X_test_expanded, y_train_expanded, y_test_expanded = train_test_split(X_expanded, y, test_size=0.2, random_state=42)
# Fit a linear regression model
model_expanded = LinearRegression()
model_expanded.fit(X_train_expanded, y_train_expanded)
# Make predictions
y_pred_expanded = model_expanded.predict(X_test_expanded)
# Evaluate the model
mse = mean_squared_error(y_test_expanded, y_pred_expanded)
mse_expanded = mean_squared_error(y_test_expanded, y_pred_expanded)
print(f'Mean Squared Error: {mse} vs Mean Squared Error Expanded: {mse_expanded}')
print(df_expanded)