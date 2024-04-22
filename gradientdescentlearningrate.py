import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
x = np.random.rand(100, 1)
y = 2 * x + np.random.rand(100, 1)

# Define the gradient descent function
def gradient_descent(x, y, learning_rate, num_iterations):
    n = x.shape[0]
    theta = np.zeros((2, 1))

    # Create a figure to plot the cost function
    costs = []
    for i in range(num_iterations):
        theta_0, theta_1 = theta
        gradient_0 = np.sum((theta_0 + theta_1 * x - y)) / n
        gradient_1 = np.sum((theta_0 + theta_1 * x - y) * x) / n
        theta_0 = theta_0 - learning_rate * gradient_0
        theta_1 = theta_1 - learning_rate * gradient_1
        theta = np.array([[theta_0], [theta_1]])
        costs.append(np.sum(np.square(theta_0 + theta_1 * x - y)) / (2 * n))

    # Plot the cost function
    plt.plot(costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()

# Run the gradient descent function with different learning rates
gradient_descent(x, y, 0.01, 100)
gradient_descent(x, y, 0.1, 100)
gradient_descent(x, y, 1, 100)