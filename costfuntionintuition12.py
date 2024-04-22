import numpy as np
x_train=np.array([1.0,2.0,3.0,4.0,5.0,6.0])
y_train=np.array([120,1000,3434,600,700,800])
def compute_cost(X, y, theta):  
    m = len(y)
    predictions = X.dot(theta)  
    errors = predictions - y    
    squared_errors = errors ** 2  
    J = 1 / (2 * m) * np.sum(squared_errors)  
    return J
X = np.column_stack((np.ones(len(x_train)), x_train))
theta = np.zeros(2)
cost = compute_cost(X, y_train, theta)
print("Initial Cost:", cost)