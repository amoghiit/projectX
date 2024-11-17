import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Add a column of ones for the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        # Compute the coefficients using the normal equation: (X'X)^-1 X'y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def predict(self, X):
        # Add a column of ones to include the intercept term
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ np.r_[self.intercept, self.coefficients]  # Concatenate intercept with coefficients
