# Gradient Descent Algorithms for Linear Regression

This repository contains implementations of Batch Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent for solving linear regression problems in Python. These methods are used to optimize the parameters of a linear regression model to minimize the mean squared error.

## Features

	•	Hypothesis Function: The model is based on the hypothesis for linear regression using a vectorized form.
	•	Cost Function (MSE): The cost function used is the Mean Squared Error (MSE) to evaluate the performance of the model.
	•	Batch Gradient Descent: Updates the parameters using the average gradient over the entire dataset.
	•	Stochastic Gradient Descent: Updates the parameters after each individual example, allowing faster, but noisier updates.
	•	Mini-Batch Gradient Descent: Combines the benefits of Batch and Stochastic Gradient Descent by updating the parameters on small subsets of the data.
## Usage

	1.	Clone the repository:

git clone https://github.com/yourusername/gradient-descent-linear-regression.git
cd gradient-descent-linear-regression

	2.	Run the script:

python gradient_descent.py

## Parameters

	•	X: Input features with a bias term added as the first column.
	•	y: Target values (the variable you are trying to predict).
	•	theta: Initial model parameters (weights).
	•	alpha: Learning rate for the gradient descent updates.
	•	num_iters: Number of iterations for the gradient descent process.
	•	batch_size: Number of examples used in each batch for Mini-Batch Gradient Descent.

## Example

The script demonstrates the use of the gradient descent algorithms on a small housing price dataset:

X = np.array([[1, 1650], [1, 896], [1, 1329], [1, 2110]])  # Features with bias term
y = np.array([215, 105, 172, 244])  # Target variable

The model is initialized with random parameters (theta), and the script will train the model using the three gradient descent methods.

## Output

The script prints the final optimized parameters (theta) for each gradient descent method:

Theta (Batch GD): [0.250 0.600]
Theta (SGD): [0.255 0.610]
Theta (Mini-Batch GD): [0.253 0.605]

## Customization

	•	You can modify the learning rate (alpha), number of iterations (num_iters), and batch size (batch_size) in the script to experiment with different settings.
	•	The normalize_features() function can be used to standardize the features for better performance during training.