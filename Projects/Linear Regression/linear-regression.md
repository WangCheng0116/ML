# Cost Function
``` python
import numpy as np
def mean_squared_error(y_true, y_pred):
    '''
    Calculate mean squared error between y_pred and y_true.
    Parameters
    ----------
    y_true (np.ndarray) : (m, 1) numpy matrix consists of true values
    y_pred (np.ndarray) : (m, 1) numpy matrix consists of predictions
    
    Returns
    -------
    The mean squared error value.
    '''
    m = len(y_true)  
    squared_errors = np.square(y_true - y_pred)
    mse = sum(squared_errors) / (2 * m)
    return mse
```

# Gradient Descent

It has m data samples, each row of X is a data sample, each column of X is a feature.  
 
``` python
def gradient_descent_multi_variable(X, y, lr=1e-5, number_of_epochs=250):
    '''
    Approximate bias and weights that give the best fitting line using gradient descent.
    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    number_of_epochs (int) : Number of gradient descent epochs
    
    Returns
    -------
    bias (float):
        The bias constant
    weights (np.ndarray):
        A (n, 1) numpy matrix that specifies the weight constants.
    loss (list):
        A list where the i-th element denotes the MSE score at i-th epoch.
    '''
    m, n = X.shape
    bias = 0
    weights = np.full((n, 1), 0.0) 
    loss = []
    for _ in range(number_of_epochs):
        predictions = X @ weights + bias
        # we can directly use bias since it is a scalar, and it will be broadcasted to the shape of predictions
        error = predictions - y
        
        bias_prime = bias - lr * np.sum(error) / m
        weights_prime = weights - lr * X.T @ (error) / m
        
        bias = bias_prime
        weights = weights_prime
        
        predictions = X @ weights + bias
        loss.append(mean_squared_error(y, predictions))
    return bias, weights, loss
```

# Normal Equation
``` python
def normal_equation(X, y):
    '''
    Approximate bias and weights that give the best fitting line using normal equation.
    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    
    Returns
    -------
    bias (float):
        The bias constant
    weights (np.ndarray):
        A (n, 1) numpy matrix that specifies the weight constants.
    '''
    # this add the bias column to the feature matrix
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    # the math formula is (X^T X)^-1 @ X^T @ y
    bias = theta[0]
    weights = theta[1:]
    return bias, weights
```

# Feature Scaling
``` python
def feature_scaling(X):
    '''
    Scale the features of X.
    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    
    Returns
    -------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    '''
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X
```