# Mini-batch Gradient Descent
- **(Batch) Gradient Descent**:
> It uses all sample data

``` python
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    a, caches = forward_propagation(X, parameters)
    cost = compute_cost(a, Y)
    grads = backward_propagation(a, caches, parameters)
    parameters = update_parameters(parameters, grads)
        
```

- **Stochastic Gradient Descent**:
> It takes only one sample data


```python
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        a, caches = forward_propagation(X[:,j], parameters) # only use one column of X
        cost = compute_cost(a, Y[:,j])
        grads = backward_propagation(a, caches, parameters)
        parameters = update_parameters(parameters, grads)
```

- **Mini-batch Gradient Descent**:
> It's the implementation in between of Batch GD and Stochastic GD

```
logic:
1. shuffle data set
2. partition (using mini-batch size is power of 2)
3. training
```

```python
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    np.random.seed(seed)           
    m = X.shape[1]   # number of training samples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m)) # Y needs to be shuffled correspndingly to match X

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1)*mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0: # not divisible
        mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size: ] # From number of batches * batch size to last
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size: ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches
```

# Momentum

Formula:
$$ \begin{cases}
v_{dW^{[l]}} = \beta v_{dW^{[l]}} + (1 - \beta) dW^{[l]} \\
W^{[l]} = W^{[l]} - \alpha v_{dW^{[l]}}
\end{cases}$$

$$\begin{cases}
v_{db^{[l]}} = \beta v_{db^{[l]}} + (1 - \beta) db^{[l]} \\
b^{[l]} = b^{[l]} - \alpha v_{db^{[l]}} 
\end{cases}$$

1. Initialize the velocity. $v_{dW^{[l]}} = 0$, $v_{db^{[l]}} = 0$, where the size of $v_{dW^{[l]}}$ is same as $W^{[l]}$ and the size of $v_{db^{[l]}}$ is same as $b^{[l]}$.  
2. Update parameters accordingly  
```python
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural networks
    for l in range(L):
        v['dW' + str(l+1)] = beta * v['dW' + str(l+1)] + (1-beta)*(grads['dW' + str(l+1)])
        v['db' + str(l+1)] = beta * v['db' + str(l+1)] + (1-beta)*(grads['db' + str(l+1)])
        
        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * v['dW' + str(l+1)]
        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * v['db' + str(l+1)]
    return parameters, v
```

**How do you choose $\beta$?**

- The larger the momentum $\beta$ is, the smoother the update because the more we take the past gradients into account. But if $\beta$ is too big, it could also smooth out the updates too much. 
- Common values for $\beta$ range from 0.8 to 0.999. If you don't feel inclined to tune this, $\beta = 0.9$ is often a reasonable default. 
- Tuning the optimal $\beta$ for your model might need trying several values to see what works best in term of reducing the value of the cost function $J$. 


# Adam
For each mini-batch, compute dW and db. At the t-th iteration:

$$v_{dW} = \beta_1 v_{dW} + (1 - \beta_1) dW$$
$$v_{db} = \beta_1 v_{db} + (1 - \beta_1) db$$
$$s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) (dW)^2$$
$$s_{db} = \beta_2 s_{db} + (1 - \beta_2) (db)^2$$

Usually, when using the Adam algorithm, bias correction is applied:

$$v^{corrected}_{dW} = \frac{v_{dW}}{1-\beta_1^t}$$
$$v^{corrected}_{db} = \frac{v_{db}}{1-\beta_1^t}$$
$$s^{corrected}_{dW} = \frac{s_{dW}}{1-\beta_2^t}$$
$$s^{corrected}_{db} = \frac{s_{db}}{1-\beta_2^t}$$

So, when updating W and b, you have:

$$W := W - \alpha \frac{v^{corrected}_{dW}}{{\sqrt{s^{corrected}_{dW}} + \epsilon}}$$

$$b := b - \alpha \frac{v^{corrected}_{db}}{{\sqrt{s^{corrected}_{db}} + \epsilon}}$$

$\epsilon$ is a small number to avoid division by zero, usually set to $10^{-8}$, and square and square root are element-wise operations.  

1. Initialize the velocity. $v_{dW^{[l]}} = 0$, $v_{db^{[l]}} = 0$, $s_{dW^{[l]}} = 0$ and $s_{db^{[l]}} = 0$, where the size of $v_{dW^{[l]}}$ and $s_{dW^{[l]}}$ is same as $W^{[l]}$ and the size of $v_{db^{[l]}}$ and $s_{db^{[l]}}$ is same as $b^{[l]}$.
2. update
> t is not a hyperparameter in the Adam optimization algorithm. Instead, it's a variable used to keep track of the number of iterations or time steps during training. 
```python
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2 # number of layers
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary
    
    for l in range(L):
        v["dW" + str(l + 1)] = beta1*v["dW" + str(l + 1)] +(1-beta1)*grads['dW' + str(l+1)]
        v["db" + str(l + 1)] = beta1*v["db" + str(l + 1)] +(1-beta1)*grads['db' + str(l+1)]

        # Compute bias-corrected first moment estimate. 
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)]/(1-(beta1)**t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)]/(1-(beta1)**t)

        # Moving average of the squared gradients. 
        s["dW" + str(l + 1)] =beta2*s["dW" + str(l + 1)] + (1-beta2)*(grads['dW' + str(l+1)]**2)
        s["db" + str(l + 1)] = beta2*s["db" + str(l + 1)] + (1-beta2)*(grads['db' + str(l+1)]**2)

        # Compute bias-corrected second raw moment estimate. 
        s_corrected["dW" + str(l + 1)] =s["dW" + str(l + 1)]/(1-(beta2)**t)
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)]/(1-(beta2)**t)

        # Update parameters. 
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)]-learning_rate*(v_corrected["dW" + str(l + 1)]/np.sqrt( s_corrected["dW" + str(l + 1)]+epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)]-learning_rate*(v_corrected["db" + str(l + 1)]/np.sqrt( s_corrected["db" + str(l + 1)]+epsilon))
        
    return parameters, v, s
```

# Summary
main logic:
```python
# Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
seed = seed + 1
minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
for minibatch in minibatches:            
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)
            # Compute cost
            cost = compute_cost(a3, minibatch_Y)
            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1     # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
```