import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
x = np.random.rand(100, 1)
y = np.zeros(100)
for i in range(50):
    y[i] = 1

# Define the logistic regression function
def logistic_regression(x, w, b):
    z = np.dot(x, w) + b
    h = 1 / (1 + np.exp(-z))
    return h

# Define the cost function
def compute_cost(x, y, w, b):
    m = len(x)
    h = logistic_regression(x, w, b)
    J = -np.sum(y * np.log(h) + (1-y) * np.log(1-h)) / m
    return J

# Initialize weights and bias
w = np.zeros((1, 1))
b = 0

# Set learning rate and number of iterations
alpha = 0.1
iterations = 1000

# Perform gradient descent
for i in range(iterations):
    h = logistic_regression(x, w, b)
    dw = np.dot(x.T, (h - y)) / len(x)
    db = np.sum(h - y) / len(x)
    w = w - alpha * dw
    b = b - alpha * db

# Plot the data points and the regression line
plt.scatter(x[:, 0], y, cmap='viridis')
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = 0, 1.5
xx = np.linspace(x_min, x_max, 100)
yy = logistic_regression(xx[:, np.newaxis], w, b)
plt.plot(xx, yy, 'k-')

# Set the plot title and labels
plt.title("Logistic Regression")
plt.xlabel("Feature 1")
plt.ylabel("Probability")

# Display the plot
plt.show()