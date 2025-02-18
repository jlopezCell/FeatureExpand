import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from featureexpand.feature_expander import FeatureExpander
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

num_data_points = 1000
expansion_depth = 3
polynomial_degree = 3

# Generate step function data - NEW DATASET
np.random.seed(42)
X = np.linspace(-5, 5, num_data_points).reshape(-1, 1)
y = np.where(X < 0, -1, 1) + np.random.normal(0, 0.2, num_data_points).reshape(-1, 1) # Step function with noise

# Normalize data between 0 and 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = pd.DataFrame(scaler_X.fit_transform(X), columns=['X'])
y = pd.DataFrame(scaler_y.fit_transform(y), columns=['y'])

print(X)
print(y)

def selection_function(x_value):
    return x_value > 0.5

def fit_and_expand_features(X_data, y_data, expansion_depth_param, feature_list, selection_function_param ):
    expander = FeatureExpander("Tp6uxDgDHf+meUtDirx0veUq7L59a6M7IsxjRqUJZlc=", enviroment="TEST")
    y_numeric = y_data.astype(float)
    cluster_labels = y_numeric.applymap(lambda yx: "x1" if selection_function_param(yx) else "x0")
    cluster_labels.columns = ["Cluster"]
    expander.fit(X_data, cluster_labels, feacture_selection=feature_list, deep=expansion_depth_param, response="x1")
    return expander

feature_expander = fit_and_expand_features(X, y, expansion_depth, ["X"], selection_function)
X_expanded = feature_expander.add_features(X)

linear_model_expand = LinearRegression()
linear_model_expand.fit(X_expanded, y)
y_linear_pred_expand = linear_model_expand.predict(X_expanded)

linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

poly_features = PolynomialFeatures(degree=polynomial_degree)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

poly_features_expanded = PolynomialFeatures(degree=polynomial_degree)
X_poly_expanded = poly_features.fit_transform(X_expanded)
poly_model_expanded = LinearRegression()
poly_model_expanded.fit(X_poly_expanded, y)
y_poly_pred_expanded = poly_model_expanded.predict(X_poly_expanded)


linear_r2 = r2_score(y, y_linear_pred)
linear_r2_expanded = r2_score(y, y_linear_pred_expand)
poly_r2 = r2_score(y, y_poly_pred)
poly_r2_expanded = r2_score(y, y_poly_pred_expanded)


def evaluate_r2_score(r2_score_value):
    if r2_score_value >= 0.9:
        return "Very good"
    elif 0.7 <= r2_score_value < 0.9:
        return "Good"
    elif 0.5 <= r2_score_value < 0.7:
        return "Fair"
    elif 0.3 <= r2_score_value < 0.5:
        return "Bad"
    else:
        return "Very bad"


linear_evaluation = evaluate_r2_score(linear_r2)
linear_evaluation_expanded = evaluate_r2_score(linear_r2_expanded)
poly_evaluation = evaluate_r2_score(poly_r2)
poly_evaluation_expanded = evaluate_r2_score(poly_r2_expanded)

print(f"Linear Regression R² score: {linear_r2:.4f} - {linear_evaluation}")
print(f"Linear Regression Expanded R² score: {linear_r2_expanded:.4f} - {linear_evaluation_expanded}")
print(f"Polynomial Regression R² score: {poly_r2:.4f} - {poly_evaluation}")
print(f"Polynomial Regression Expanded R² score: {poly_r2_expanded:.4f} - {poly_evaluation_expanded}")

plt.figure(figsize=(12, 6))
plt.scatter(X.values.flatten(), y.values.flatten(), color='blue', alpha=0.5, label='Original Data (Step Function)') # Updated label
plt.plot(X, y_linear_pred, color='red', label='Linear Regression')
plt.plot(X, y_linear_pred_expand, color='blue', label='Expanded Linear Regression')
plt.plot(X, y_poly_pred, color='green', label='Polynomial Regression')
plt.plot(X, y_poly_pred_expanded, color='purple', label='Expanded Polynomial Regression')
plt.title('Example: Linear Regression Failing on Step Function Data') # Updated title
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()