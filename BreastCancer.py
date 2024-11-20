import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 for malignant, 1 for benign

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Standardize features
y = y.reshape(-1, 1)  # Reshape y to be a column vector

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize parameters
def initialize_params(n_features):
    weights = np.zeros((n_features, 1))
    bias = 0
    return weights, bias

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Compute cost and gradients
def compute_cost_and_gradients(X, y, weights, bias):
    m = X.shape[0]
    z = np.dot(X, weights) + bias
    A = sigmoid(z)
    cost = -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    dw = (1 / m) * np.dot(X.T, (A - y))
    db = (1 / m) * np.sum(A - y)
    return cost, dw, db

# Gradient descent optimization
def gradient_descent(X, y, weights, bias, learning_rate, num_iterations):
    for i in range(num_iterations):
        cost, dw, db = compute_cost_and_gradients(X, y, weights, bias)
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.4f}")

    return weights, bias

# Make predictions
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    A = sigmoid(z)
    return (A > 0.5).astype(int)

# Train the model
n_features = X_train.shape[1]
weights, bias = initialize_params(n_features)
learning_rate = 0.01
num_iterations = 1000

weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate, num_iterations)

# Evaluate the model
y_train_pred = predict(X_train, weights, bias)
y_test_pred = predict(X_test, weights, bias)

print("\nTraining Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
