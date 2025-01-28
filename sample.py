import pandas as pd
from featureexpand.feature_expander import FeatureExpander
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a sample DataFrame
data = {
    'A': [0, 1, 1, 1, 0, 1, 1],
    'B': [1, 0, 1, 0, 0, 0, 1],
    'Cluster': [1, 1, 0, 1, 0, 1, 0]  # Target variable
}

df = pd.DataFrame(data)

print(df)
# Split the data into training and testing sets
X = df.drop(columns=['Cluster'])
y = df['Cluster']

print(X,"xxx", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

precision = 1
# Initialize the FeatureExpander
expander = FeatureExpander()
# Add new features
yy=pd.Series({'Cluster': ["x1", "x1", "x0", "x1", "x0", "x1", "x0"]})
X_expanded = expander.add_features(X,yy,precision)
values = [1,0]
print("Resultados ",expander.transform(values))
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