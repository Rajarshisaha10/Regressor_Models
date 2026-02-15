import numpy as np
import matplotlib.pyplot as plt

# 1. Feature Scaling
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

# 2. Ridge Regression Train Function
def train_ridge_regression(X, y, learning_rate=0.01, epochs=1000, lambda_param=0.1):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []

    for epoch in range(epochs):
        # Predictions
        y_pred = np.dot(X, weights) + bias
        error = y_pred - y
        
        # Loss Calculation: Mean Squared Error + L2 Penalty
        mse_loss = np.mean(error ** 2)
        l2_penalty = lambda_param * np.sum(np.square(weights))
        total_loss = mse_loss + l2_penalty
        loss_history.append(total_loss)

        # Gradients (including L2 derivative: 2 * lambda * weights)
        dw = (2 / n_samples) * np.dot(X.T, error) + (2 * lambda_param * weights)
        db = (2 / n_samples) * np.sum(error)

        # Gradient Descent Update
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, loss_history

# 3. Prediction Function
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

# MAIN EXECUTION

# Example Dataset: [Square Footage, Bedrooms]
X = np.array([
    [2100, 3],
    [1600, 3],
    [2400, 4],
    [1400, 2],
    [3000, 4]
], dtype=float)

# Prices in $k
y = np.array([400, 330, 460, 280, 520], dtype=float)

# Step 1: Normalize
X_scaled, mean, std = normalize_features(X)

# Step 2: Train Model
learning_rate = 0.1
epochs = 500
lambda_val = 0.1 # Regularization strength

weights, bias, loss_history = train_ridge_regression(
    X_scaled, y, learning_rate, epochs, lambda_val
)

# Step 3: Predict for a new custom sample
new_house = np.array([2000, 3])
new_house_scaled = (new_house - mean) / std
prediction = predict(new_house_scaled, weights, bias)

print(f"Weights: {weights}")
print(f"Bias: {bias:.2f}")
print(f"Predicted Price for 2000sqft, 3BR: ${prediction:.2f}k")

