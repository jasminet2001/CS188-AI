from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow import keras
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# code for the second and third part
# part 1
# Define the functions
def linear_function(x):
    return 2*x + 5


def quadratic_function(x):
    return x**2 + 3*x - 2


def sine_function(x):
    return np.sin(x)


# Generate sample points for functions
X_linear = np.linspace(0, 10, 50)
y_linear = linear_function(X_linear)

X_quadratic = np.linspace(-5, 5, 50)
y_quadratic = quadratic_function(X_quadratic)

X_sine = np.linspace(-np.pi, np.pi, 50)
y_sine = sine_function(X_sine)

X = np.concatenate((X_linear, X_quadratic, X_sine))
y = np.concatenate((y_linear, y_quadratic, y_sine))

# Define, train, generate the MLP
mlp = MLPRegressor(hidden_layer_sizes=(
    100,), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X.reshape(-1, 1), y)
X_test = np.linspace(-2*np.pi, 2*np.pi, 100)
y_test = sine_function(X_test)
y_predicted = mlp.predict(X_test.reshape(-1, 1))

plt.plot(X_test, y_test, label='Correct Function')
plt.plot(X_test, y_predicted, label='Predicted Function')
plt.plot(X_test, abs(y_test - y_predicted), label='Error Rate')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_predicted)
mae = mean_absolute_error(y_test, y_predicted)

# part 2
# Add random noise to the sample points
y_linear_noisy = y_linear + np.random.normal(0, 1, size=len(y_linear))
y_quadratic_noisy = y_quadratic + np.random.normal(0, 5, size=len(y_quadratic))
y_sine_noisy = y_sine + np.random.normal(0, 0.1, size=len(y_sine))

# Combine the sample points into a single dataset
X = np.concatenate((X_linear, X_quadratic, X_sine))
y = np.concatenate((y_linear_noisy, y_quadratic_noisy, y_sine_noisy))
y_test_noisy = y_test + np.random.normal(0, 0.1, size=len(y_test))

plt.plot(X_test, y_test_noisy, label='Correct Function')
plt.plot(X_test, y_predicted, label='Predicted Function')
plt.plot(X_test, abs(y_test_noisy - y_predicted), label='Error Rate')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()

mse = mean_squared_error(y_test_noisy, y_predicted)
mae = mean_absolute_error(y_test_noisy, y_predicted)

print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

# part2,3
# Define the functions
def linear_function(X):
    return 2*X[:, 0] + 3*X[:, 1] + 5


def quadratic_function(X):
    return X[:, 0]**2 + 3*X[:, 1]**2 + 2*X[:, 0]*X[:, 1] + 4*X[:, 0] - 2*X[:, 1] - 1


def sine_function(X):
    return np.sin(X[:, 0]) + np.cos(X[:, 1])


# Define the range and number of sample points
x_range = (-5, 5)
y_range = (-5, 5)
n_samples = 50

mlp = MLPRegressor(hidden_layer_sizes=(
    100,), activation='relu', solver='adam', max_iter=1000)
for func_name, func in zip(['linear', 'quadratic', 'sine'], [linear_function, quadratic_function, sine_function]):

    X = np.random.uniform(x_range[0], x_range[1], size=(n_samples, 2))
    y = func(X)

    for noise_level in [0.1, 0.5, 1.0]:

        X_noisy = X + noise_level * np.random.randn(*X.shape)
        y_noisy = y + noise_level * np.random.randn(*y.shape)
        mlp.fit(X_noisy, y_noisy)
        X_test = np.random.uniform(x_range[0]-1, x_range[1]+1, size=(10000, 2))
        y_test = func(X_test)
        y_predicted = mlp.predict(X_test)

        mse = mean_squared_error(y_test, y_predicted)
        mae = mean_absolute_error(y_test, y_predicted)

        plt.figure()
        plt.suptitle(
            f'{func_name.capitalize()} Function, Noise Level {noise_level}')
        plt.subplot(1, 3, 1)
        plt.title('Function')
        plt.scatter(X[:, 0], y, label='Sample Points')
        plt.plot(X_test[:, 0], y_test, label='True Function')
        plt.xlim(x_range)
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.title('Network Output')
        plt.scatter(X_noisy[:, 0], y_noisy, label='Noisy Sample Points')
        plt.plot(X_test[:, 0], y_predicted, label='Network Output')
        plt.xlim(x_range)
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.title('Error')
        plt.plot(X_test[:, 0], y_test - y_predicted, label='Error')
        plt.xlim(x_range)
        plt.legend()
        plt.tight_layout()

        print(f'{func_name.capitalize()} Function, Noise Level {noise_level}')
        print('Mean Squared Error:', mse)
        print('Mean Absolute Error:', mae)

plt.show()
