import numpy as np
import matplotlib.pyplot as plt

# Function to minimize (cost function)
def cost_function(theta):
    return theta ** 2  # Example cost function (squared error)

# Gradient of the cost function
def gradient(theta):
    return 2 * theta  # Gradient of the squared error function

# Gradient descent function
def gradient_descent(learning_rate=0.1, n_iterations=10):
    theta = 4  # Initial guess for parameter
    theta_history = [theta]
    cost_history = [cost_function(theta)]
    for iteration in range(n_iterations):
        gradient_value = gradient(theta)
        theta = theta - learning_rate * gradient_value  # Update parameter using gradient
        theta_history.append(theta)
        cost_history.append(cost_function(theta))
    return theta_history, cost_history

# Perform gradient descent
learning_rate = 0.1
n_iterations = 10
theta_history, cost_history = gradient_descent(learning_rate, n_iterations)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(theta_history, cost_history, marker='o', linestyle='-', color='b')
plt.xlabel('Parameter (Theta)')
plt.ylabel('Cost')
plt.title('Gradient Descent Intuition')
plt.grid(True)
plt.show()
