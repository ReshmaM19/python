import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
x = np.random.rand(100, 2)
y = np.zeros(100)
for i in range(50):
    if np.random.rand() < 0.5:
        y[i] = 1
x = np.hstack((x, np.ones((100, 1))))

# Define the logistic regression function
def logistic_regression(x, w):
    return 1 / (1 + np.exp(-np.dot(x, w)))

# Define the cost function
def compute_cost(x, y, w):
    m = len(x)
    h = logistic_regression(x, w)
    J = -np.sum(y * np.log(h) + (1-y) * np.log(1-h)) / m
    return J

# Define the gradient function
def compute_gradient(x, y, w):
    m = len(x)
    h = logistic_regression(x, w)
    grad = np.dot(x.T, (h - y)) / m
    return grad

# Initialize weights
w = np.zeros((3, 1))

# Set learning rate and number of iterations
alpha = 0.1
iterations = 1000

# Perform gradient descent
for i in range(iterations):
    grad = compute_gradient(x, y, w)
    w = w - alpha * grad

# Calculate decision boundary
x1 = np.linspace(-0.5, 1.5, 100)
x2 = (-w[2] - w[0] * x1) / w[1]

# Plot the data points and decision boundary
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
plt.plot(x1, x2, 'k-')
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# Set the plot title and labels
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Display the plot
plt.show()