import numpy as np
import matplotlib.pyplot as plt

# Define the logistic loss function
def logistic_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Generate some random data
x = np.random.rand(100, 1)
y = np.zeros(100)
for i in range(50):
    y[i] = 1
y = y.reshape(-1, 1)

# Define the logistic regression function
def logistic_regression(x, w):
    return 1 / (1 + np.exp(-np.dot(x, w)))

# Initialize weights
w = np.zeros((1, 1))

# Set learning rate and number of iterations
alpha = 0.1
iterations = 100

# Perform gradient descent
for i in range(iterations):
    y_pred = logistic_regression(x, w)
    loss = logistic_loss(y, y_pred)
    grad = np.dot(x.T, (y_pred - y)) / len(x)
    w = w - alpha * grad

# Plot the logistic loss vs. iteration number
plt.plot(range(iterations), [logistic_loss(y, logistic_regression(x, w)) for w in np.array([w for _ in range(iterations)])])
plt.xlabel("Iteration")
plt.ylabel("Logistic Loss")
plt.title("Logistic Regression Logistic Loss vs. Iteration")
plt.ylim(0, 1)
plt.xlim(0, iterations)

# Display the plot
plt.show()