import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


np.random.seed(0)
X = 2 * np.random.rand(100, 1) 
y = 5 * X**2 + np.random.randn(100, 1)  


poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)


model = LinearRegression()
model.fit(X_poly, y)


X_new = np.linspace(0, 2, 100).reshape(-1, 1)
X_new_poly = poly_features.transform(X_new)
y_pred = model.predict(X_new_poly)


plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Original Data')


plt.plot(X_new, y_pred, color='red', label='Predicted Values')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Feature Engineering: Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
