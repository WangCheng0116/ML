## For NN, we would organize input where each column is a data

## feature scaling HELPS A LOOOOT!!!
``` python
mean = np.mean(X_train, axis=0)
std_deviation = np.std(X_train, axis=0)
X_train = (X_train - mean) / std_deviation
```

## Initialization -> use np.random.rand(m, n) * 0.01

## Consice way for function expression
``` python
def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deri(Z):
    # Don't care about whether Z is a number or an array
    return Z > 0
```

## Main logic template
``` python
def FP(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    
def BP(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.shape[0]

    # This is our endeavor to convert an array of labels to [[0,0,0,0,0,1,0,0], ...[1,0,0,0,0...,0]]
    temp = np.zeros((Y.size, 10)) # inialize it
    temp[np.arange(Y.size), Y] = 1 # A very smart way for one hot encoding 
    temp = temp.T
   
    dZ2 = A2 - temp
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = W2.T.dot(dZ2) * ReLU_deri(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis = 1, keepdims = True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```

## One Hot Encoding
It involves creating a binary (0/1) indicator variable for each category. Each indicator variable corresponds to a category and is also called a "dummy variable." If there are n categories, n dummy variables are created. Only one of them is "hot" (1) at a time, indicating the category's presence.

For example, if you have three colors (red, green, blue), you would create three dummy variables:

red -> [1, 0, 0]
green -> [0, 1, 0]
blue -> [0, 0, 1]
