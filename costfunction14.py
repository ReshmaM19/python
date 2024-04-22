import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(0)
X = 2 * np.random.rand(100, 1) 
y = 4 + 3 * X + np.random.randn(100, 1) 


model = LinearRegression()
model.fit(X, y)


X_new = np.array([[0], [2]]) 
y_pred = model.predict(X_new)


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_new, y_pred, color='red', linewidth=2, label='Linear regression model')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Simple Linear Regression Example')
plt.legend()
plt.grid(True)
plt.show()
