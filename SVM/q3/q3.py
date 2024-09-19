from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import zipfile
import io
import os
import requests


# Load the dataset from a local CSV file
# Replace with the correct file path
file_path = './BostonHousing.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Split the dataset into features and target variable
X = df.drop(columns=['medv'])
y = df['medv']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVR model with RBF kernel
model = SVR(kernel='linear')
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='darkorange', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='navy', lw=2, linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.show()

# Plot the comparison between actual and predicted values
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual Prices', color='blue', marker='o')
plt.plot(y_pred, label='Predicted Prices', color='red', marker='x')
plt.title('Comparison of Actual and Predicted Prices')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# Cross-validation scores
scores = cross_val_score(model, X_train_scaled, y_train,
                         cv=5, scoring='neg_mean_squared_error')
print("Cross-validation MSE scores:", -scores)
print("Mean cross-validation MSE score:", -scores.mean())
