import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Random features
y = 5 * X**2 + np.random.randn(100, 1)  # Quadratic relationship with noise

# Polynomial features
degree = 10
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Regularization parameter
alpha_values = [0, 1e-6, 1e-3, 1]

# Plot original data
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Original Data')

# Plot polynomial regression with different regularization strengths
for alpha in alpha_values:
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=alpha))
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, label=f'Ridge Regression (alpha={alpha})')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Regularization to Reduce Overfitting: Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
