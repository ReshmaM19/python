import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Random features
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Add bias term to features
X_b = np.c_[np.ones((100, 1)), X]  # Add x0 = 1 to each instance

# Gradient descent function
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    theta = np.random.randn(2, 1)  # Random initialization
    m = len(X)
    for iteration in range(n_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
    return theta

# Train the model using gradient descent
theta = gradient_descent(X_b, y)

# Predictions
X_new = np.array([[0], [2]])  # New data points for prediction
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add x0 = 1 to each instance
y_pred = X_new_b.dot(theta)

# Visualize the data and the linear regression model
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_new, y_pred, color='red', linewidth=2, label='Linear regression model (Gradient Descent)')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()
