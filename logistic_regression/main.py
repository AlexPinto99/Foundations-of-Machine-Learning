import numpy as np


# Sigmoid function (hypothesis representation)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Hypothesis function for logistic regression
def hypothesis(X, theta):
    return sigmoid(np.dot(X, theta))


# Cost function (Cross Entropy / Log Loss)
def compute_cost(X, y, theta):
    m = len(y)
    h = hypothesis(X, theta)
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))


# Gradient Descent for Logistic Regression
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for _ in range(num_iters):
        h = hypothesis(X, theta)
        gradient = np.dot(X.T, (h - y)) / m
        theta = theta - alpha * gradient

        # Record the cost at each iteration
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


# Data normalization (optional)
def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Example usage of logistic regression
if __name__ == "__main__":
    # Example dataset (binary classification)
    X = np.array([[1, 34], [1, 78], [1, 64], [1, 50], [1, 90]])  # Features with bias term
    y = np.array([0, 1, 1, 0, 1])  # Target variable (0 or 1 for binary classification)

    # Initialize parameters
    theta = np.random.rand(2)  # Random initialization of theta
    alpha = 0.01  # Learning rate
    num_iters = 1000  # Number of iterations for gradient descent

    # Normalize features (excluding bias term)
    X[:, 1] = normalize_features(X[:, 1])

    # Run gradient descent for logistic regression
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iters)

    # Output the final parameters and the cost
    print("Final theta values:", theta)
    print("Final cost:", cost_history[-1])