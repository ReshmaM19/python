import numpy as np

x_train=np.array([1.0,2.0,3.0,4.0,5.0,6.0])
y_train=np.array([120,1000,3434,600,700,800])

def compute_cost(X, y, theta):
   
    m = len(y)
    predictions = X.dot(theta)
    sqr_errors = (predictions - y) ** 2
    J = 1 / (2 * m) * np.sum(sqr_errors)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
   
    m = len(y)
    J_history = []

    for iter in range(num_iters):
        predictions = X.dot(theta)
        errors = np.dot(X.transpose(), (predictions - y))
        theta -= alpha * (1 / m) * errors
        J_history.append(compute_cost(X, y, theta))
        
    return theta, J_history


X = np.column_stack((np.ones(len(x_train)), x_train))


theta = np.zeros(2)


alpha = 0.01
num_iters = 1500

theta, J_history = gradient_descent(X, y_train, theta, alpha, num_iters)


print("Theta found by gradient descent:", theta)


import matplotlib.pyplot as plt
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.title('Cost Function Convergence')
plt.show()