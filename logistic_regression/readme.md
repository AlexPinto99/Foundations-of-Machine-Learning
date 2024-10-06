# Logistic Regression in Python

This repository contains an implementation of Logistic Regression for binary classification using Gradient Descent in Python. The implementation covers the core concepts of Logistic Regression, including the hypothesis function, cost function, and parameter optimization through gradient descent.

## Features

	•	Sigmoid Hypothesis Function: Uses the sigmoid function to calculate the probability for binary classification.
	•	Cost Function: Implements Cross Entropy Loss to measure prediction errors.
	•	Gradient Descent: Iteratively updates the parameters (theta) to minimize the cost function.
	•	Feature Normalization: Ensures efficient convergence during gradient descent by scaling the feature values.
	•	Binary Classification: Demonstrates classification for datasets with two classes (e.g., 0 and 1).

## Usage

1.	Clone the repository:

```bash
git clone https://github.com/yourusername/logistic-regression.git
cd logistic-regression
```

2.	Run the script:

```bash
python logistic_regression.py
```
By default, the script uses a small sample dataset. You can replace this dataset with your own binary classification data by editing the X and y variables in logistic_regression.py.

## Example

The script runs a logistic regression model on a small dataset of features and binary labels:

```python
X = np.array([[1, 34], [1, 78], [1, 64], [1, 50], [1, 90]])  # Features with bias term
y = np.array([0, 1, 1, 0, 1])  # Binary target variable (0 or 1)
```
## Output

After training the model using gradient descent, the script will output the optimized parameters (theta) and the final cost:

```bash
Final theta values: [0.3462 1.2795]
Final cost: 0.4731
```

## Customization

	•	You can modify the learning rate (alpha), number of iterations (num_iters), and dataset to experiment with different models and improve the performance for your own data.
	•	The normalize_features() function helps normalize the feature values, which can improve convergence speed.
