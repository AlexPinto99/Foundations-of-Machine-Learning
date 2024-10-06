import numpy as np

# Sample training data
# Feature: Size of house (1 variable)
X = np.array([1500, 2000, 2500, 3000, 3500])
y = np.array([300000, 400000, 500000, 600000, 700000])

# Number of training examples
m = len(y)

# Initialize parameters
theta_0 = 0  # intercept
theta_1 = 0  # slope
alpha = 0.00000001  # learning rate
iterations = 1000  # number of iterations

# Gradient descent
for _ in range(iterations):
    # Compute the predictions for all training examples
    predictions = theta_0 + theta_1 * X

    # Calculate the gradients
    gradient_0 = (1 / m) * np.sum(predictions - y)  # Gradient for theta_0
    gradient_1 = (1 / m) * np.sum((predictions - y) * X)  # Gradient for theta_1

    # Update the parameters
    theta_0 -= alpha * gradient_0  # Update theta_0 (intercept)
    theta_1 -= alpha * gradient_1  # Update theta_1 (slope)

# Final parameters
print(f"theta_0 (intercept): {theta_0}")
print(f"theta_1 (slope): {theta_1}")

# To predict the price of a new house
new_size = 3000  # Example: 3000 sq ft house
predicted_price = theta_0 + theta_1 * new_size
print(f"Predicted price: {predicted_price}")