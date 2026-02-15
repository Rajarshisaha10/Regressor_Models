import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(10)

X = 2 * np.random.rand(800)
noise = np.random.randn(800)
Y = 4 + 3 * X + noise


m = 0.0   # initializing slope to zero
c = 0.0   # initializing intercept to zero

learning_rate = 0.01
epochs = 1000
n = len(X)

losses = []

for epoch in range(epochs):

    # Predictions
    Y_pred = m * X + c

    # Calculating Mean Square Error (MSE)
    loss = np.mean((Y - Y_pred) ** 2)
    losses.append(loss)

    # Calculationg Gradients
    dm = (-2 / n) * np.sum(X * (Y - Y_pred))
    dc = (-2 / n) * np.sum(Y - Y_pred)

    # Update parameters after calculations
    m = m - learning_rate * dm
    c = c - learning_rate * dc

    # Print occasionally
    # if epoch % 100 == 0:
    print(f"Epoch {epoch}: Loss={loss:.4f}, m={m:.4f}, c={c:.4f}")
    time.sleep(0.3)


print(f"\nFinal Model: y = {m:.2f}x + {c:.2f}")


# Predictions using final parameters
Y_final_pred = m * X + c

# Sum of Squares Residuals (SSR)
ssr = np.sum((Y - Y_final_pred) ** 2)

# Total Sum of Squares (SST)
sst = np.sum((Y - np.mean(Y)) ** 2)

r2_score = 1 - (ssr / sst)
print(f"R-squared Score: {r2_score:.4f}")


plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_final_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Model Fit')
plt.show()


mae = np.mean(np.abs(Y - Y_final_pred))
print(f"Mean Absolute Error: {mae:.4f}")
