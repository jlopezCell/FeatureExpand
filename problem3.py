import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from featureexpand.feature_expander import FeatureExpander # 🚀 Our interpretability engine!

num_data_points = 1000
expansion_depth = 4
polynomial_degree = 3

# TensorFlow imports (Conceptual - for demonstration)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

num_data_points = 1000
forecast_horizon = 1 # Predict the next time step

# Generate time series data with seasonality and trend - NEW DATASET
np.random.seed(42)
time = np.arange(num_data_points)
amplitude = np.sin(time / 50) * 5 + time / 100  # Seasonality + Trend
noise = np.random.normal(0, 0.5, num_data_points)
y_time_series = amplitude + noise
X_time_series = time.reshape(-1, 1)

# Normalize data between 0 and 1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X = pd.DataFrame(scaler_X.fit_transform(X_time_series), columns=['Time']) # Renamed to 'Time'
y = pd.DataFrame(scaler_y.fit_transform(y_time_series.reshape(-1, 1)), columns=['Value']) # Renamed to 'Value'

def selection_function(x_value):
    return x_value > 0.5

def fit_and_expand_features(X_data, y_data, expansion_depth_param, feature_list, selection_function_param ):
    expander = FeatureExpander("Tp6uxDgDHf+meUtDirx0veUq7L59a6M7IsxjRqUJZlc=", enviroment="TEST") # ✨ FeatureExpander in action!
    y_numeric = y_data.astype(float)
    cluster_labels = y_numeric.applymap(lambda yx: "x1" if selection_function_param(yx) else "x0")
    cluster_labels.columns = ["Cluster"]
    expander.fit(X_data, cluster_labels, feacture_selection=feature_list, deep=expansion_depth_param, response="x1") # 🧠 Boolean simplification-based learning!
    return expander

feature_expander = fit_and_expand_features(X, y, expansion_depth, ["Time"], selection_function)
X_expanded = feature_expander.add_features(X) # 🪄 Interpretable feature generation!

print(X.head())
print(y.head())

# No FeatureExpander needed for this conceptual example - focusing on model type


linear_model_time_series = LinearRegression()
linear_model_time_series.fit(X, y) # Using Time as the feature
y_linear_pred_time_series = linear_model_time_series.predict(X)

linear_model_time_series_expanded = LinearRegression()
linear_model_time_series_expanded.fit(X_expanded, y) # Using Time as the feature
y_linear_pred_time_series_expanded = linear_model_time_series_expanded.predict(X_expanded)

# Better approach: TensorFlow LSTM (Conceptual - showing structure, not training)
# In reality, you'd need to prepare data for LSTM input (sequences) and train
lstm_model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(None, 1)), # Input shape (timesteps, features) - Conceptual
    Dense(units=1) # Output layer for single value prediction
])
# lstm_model.compile(...) # You'd compile and train in a real scenario
# y_lstm_pred_time_series = lstm_model.predict(...) # Prediction would be sequence-based

# Calculate R² score for Linear Regression
linear_r2_time_series = r2_score(y, y_linear_pred_time_series)
linear_r2_time_series_expanded = r2_score(y, y_linear_pred_time_series_expanded)


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

linear_evaluation_time_series = evaluate_r2_score(linear_r2_time_series)
linear_evaluation_time_series_expanded = evaluate_r2_score(linear_r2_time_series_expanded)


print(f"Linear Regression R² score (Time Series): {linear_r2_time_series:.4f} - {linear_evaluation_time_series}")
print(f"Linear Regression Extended R² score (Time Series): {linear_r2_time_series_expanded:.4f} - {linear_evaluation_time_series_expanded}")
# (LSTM performance would be shown here in a full implementation)


plt.figure(figsize=(12, 6))
plt.scatter(X['Time'], y['Value'], color='blue', alpha=0.5, label='Original Time Series Data') # Updated labels
plt.plot(X['Time'], y_linear_pred_time_series, color='red', label='Linear Regression') # Updated labels
# plt.plot(X['Time'], y_lstm_pred_time_series, color='green', label='LSTM (Conceptual)') # Would plot LSTM prediction in a full implementation
plt.title('Example: Linear Regression Failing on Time Series Data') # Updated title
plt.xlabel('Time') # Updated label
plt.ylabel('Value') # Updated label
plt.legend()
plt.grid(True)
plt.show()