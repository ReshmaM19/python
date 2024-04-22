import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
num_samples = 100
num_features = 2
X = 10 * np.random.rand(num_samples, num_features)  # Random features
true_theta = np.array([3, 5])  # True coefficients
noise = np.random.randn(num_samples)  # Noise
y = X.dot(true_theta) + noise  # Linear relationship with noise

# Add bias term to features
X_b = np.c_[np.ones((num_samples, 1)), X]  # Add x0 = 1 to each instance

# Cost function (mean squared error)
def compute_cost(X, y, theta):
    m = len(y)
    J = (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)
    return J

# Gradient of the cost function
def compute_gradient(X, y, theta):
    m = len(y)
    gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
    return gradients

# Gradient descent function
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    theta = np.random.randn(num_features + 1)  # Random initialization
    J_history = []
    for iteration in range(n_iterations):
        gradients = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradients
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

# Perform gradient descent
learning_rate = 0.01
n_iterations = 1000
theta_hat, J_history = gradient_descent(X_b, y, learning_rate, n_iterations)

# Display convergence graph
plt.plot(range(1, n_iterations + 1), J_history, color='b')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Convergence')
plt.grid(True)
plt.show()
