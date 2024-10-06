This will implement the course's useful Machine Learning codes and theory.

---

# Simple Linear Regression in Python

This project demonstrates a simple implementation of **linear regression** with one feature (size of the house) using **gradient descent** in Python. The goal is to predict the price of a house based on its size (square footage).

## Features

- **Single variable linear regression**: The model uses one feature (size of the house) to predict the price.
- **Gradient Descent Algorithm**: Updates the parameters iteratively to minimize the cost function.
- **Manual calculation of gradients**: The code calculates the gradients for both the intercept (\(\theta_0\)) and slope (\(\theta_1\)) separately, for better understanding.

## Files

- `main.py`: Contains the code for training the linear regression model using gradient descent and making predictions.

## Requirements

- Python 3.x
- Numpy library

Install the required dependencies by running:

```bash
pip install numpy
```

## Code Overview

The code performs the following steps:

1. **Data Preparation**: 
   - The data consists of house sizes (`X`) and their corresponding prices (`y`).

2. **Initialization**: 
   - Initializes the parameters (`theta_0` and `theta_1`) to zero.
   - Sets the learning rate (`alpha`) and the number of iterations for gradient descent.

3. **Gradient Descent**: 
   - Iteratively updates `theta_0` (intercept) and `theta_1` (slope) using the gradient of the cost function.

4. **Prediction**:
   - After training, the model can predict the price of a new house based on its size.

## Example

Sample data:

```python
# Size of house (sq ft)
X = np.array([1500, 2000, 2500, 3000, 3500])

# Price of house ($)
y = np.array([300000, 400000, 500000, 600000, 700000])
```

Predicted house price for a 3000 sq ft house:

```bash
theta_0 (intercept): 280.964458844282
theta_1 (slope): 199.99859805350743
Predicted price: 600000.2780183666
```

## How to Run

1. Clone the repository or copy the code into a Python script.
2. Run the script using:

```bash
python main.py
```

You will see the final values for the parameters \(\theta_0\) and \(\theta_1\) along with the predicted price for a new house.

## Notes

- The learning rate is set to a very small value to ensure the gradient descent converges slowly and safely. You can experiment with different learning rates.
- The number of iterations is set to 1000. You can adjust this depending on how quickly you want the model to converge.
