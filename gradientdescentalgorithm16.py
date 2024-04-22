import numpy as np

# Define the cost function J(w, b)
def cost_function(w, b, X, y):
    m = len(y)
    J = (1 / (2 * m)) * np.sum((w * X + b - y) ** 2)
    return J

# Define the gradient of the cost function with respect to w and b
def gradient(w, b, X, y):
    m = len(y)
    dw = (1 / m) * np.sum(X * (w * X + b - y))
    db = (1 / m) * np.sum(w * X + b - y)
    return dw, db

# Define the gradient descent algorithm
def gradient_descent(X, y, w_initial, b_initial, learning_rate, num_iterations):
    w = w_initial
    b = b_initial
    for i in range(num_iterations):
        dw, db = gradient(w, b, X, y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return w, b

# Define the data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Define the initial weights and bias
w_initial = 0
b_initial = 0

# Define the learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Run the gradient descent algorithm
w, b = gradient_descent(X, y, w_initial, b_initial, learning_rate, num_iterations)

# Print the final weights and bias
print("w =", w)
print("b =", b)