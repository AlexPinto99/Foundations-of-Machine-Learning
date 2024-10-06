import numpy as np

# Hypothesis function (linear regression)
def hypothesis(X, theta):
    return np.dot(X, theta)

# Cost function (Mean Squared Error)
def compute_cost(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum((hypothesis(X, theta) - y)**2)

# Batch Gradient Descent
def batch_gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        gradient = np.dot(X.T, (hypothesis(X, theta) - y)) / m
        theta = theta - alpha * gradient
    return theta

# Stochastic Gradient Descent
def stochastic_gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    for _ in range(num_iters):
        for i in range(m):
            gradient = (hypothesis(X[i], theta) - y[i]) * X[i]
            theta = theta - alpha * gradient
    return theta

# Mini-Batch Gradient Descent
def mini_batch_gradient_descent(X, y, theta, alpha, num_iters, batch_size):
    m = len(y)
    for _ in range(num_iters):
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            gradient = np.dot(X_batch.T, (hypothesis(X_batch, theta) - y_batch)) / batch_size
            theta = theta - alpha * gradient
    return theta

# Data normalization (optional, improves performance)
def normalize_features(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Sample usage of the algorithms
if __name__ == "__main__":
    # Example dataset
    X = np.array([[1, 1650], [1, 896], [1, 1329], [1, 2110]])  # Features with bias term
    y = np.array([215, 105, 172, 244])  # Target variable
    theta = np.random.rand(2)  # Random initialization of theta

    # Normalize features
    X[:, 1] = normalize_features(X[:, 1])

    # Parameters
    alpha = 0.01  # Learning rate
    num_iters = 1000  # Number of iterations
    batch_size = 2  # For mini-batch gradient descent

    # Batch Gradient Descent
    theta_batch = batch_gradient_descent(X, y, theta, alpha, num_iters)
    print("Theta (Batch GD):", theta_batch)

    # Stochastic Gradient Descent
    theta_sgd = stochastic_gradient_descent(X, y, theta, alpha, num_iters)
    print("Theta (SGD):", theta_sgd)

    # Mini-Batch Gradient Descent
    theta_mini_batch = mini_batch_gradient_descent(X, y, theta, alpha, num_iters, batch_size)
    print("Theta (Mini-Batch GD):", theta_mini_batch)