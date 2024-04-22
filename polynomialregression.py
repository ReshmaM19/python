import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
X = 2 * np.random.rand(100, 1) 
y = 5 * X**2 + np.random.randn(100, 1) 


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Original Data')


X_poly = np.c_[X, X**2] 


plt.scatter(X_poly[:, 0], y, color='red', label='Engineered Features (X, X^2)')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Feature Engineering: Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
