import numpy as np
import matplotlib.pyplot as plt

# Define the logistic loss function
def logistic_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Define the logistic regression function
def logistic_regression(x, w):
    return 1 / (1 + np.exp(-np.dot(x, w)))

# Define the gradient of the logistic loss function
def logistic_loss_gradient(x, y_true, y_pred, w):
    return np.dot(x.T, (y_pred - y_true)) / len(x)

# Initialize weights
w = np.zeros((2, 1))

# Set learning rate and number of iterations
alpha = 0.1
iterations = 100

# Generate some random data
x = np.random.rand(100, 2)
y = np.zeros(100)
for i in range(50):
    y[i] = 1
y = y.reshape(-1, 1)

# Define the range of w values
w_min, w_max = -1, 1
W = np.arange(w_min, w_max, 0.01).reshape(-1, 1)

# Calculate the cost for each combination of w values
costs = []
for w_val in W:
    w = w_val * np.ones((2, 1))
    y_pred = logistic_regression(x, w)
    cost = logistic_loss(y, y_pred)
    costs.append(cost)

# Plot the cost function
plt.plot(W, costs)
plt.xlabel("Weight Value")
plt.ylabel("Cost")
plt.title("Cost Function for Logistic Regression")

# Display the plot
plt.show()