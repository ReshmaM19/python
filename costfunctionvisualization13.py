import numpy as np
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
   
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    squared_errors = errors ** 2
    J = 1 / (2 * m) * np.sum(squared_errors)
    return J

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 5, 4, 5])

X = np.column_stack((np.ones(len(x_train)), x_train))

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i, theta0 in enumerate(theta0_vals):
    for j, theta1 in enumerate(theta1_vals):
        theta = np.array([theta0, theta1])
        J_vals[i, j] = compute_cost(X, y_train, theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0, theta1, J_vals.T, cmap='viridis')
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
ax.set_title('Cost Function Visualization')
plt.show()