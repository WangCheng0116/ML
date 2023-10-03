**IDEA: Shut down neurons randomly**
> Forward, $A^{[l]}$ multiply with $D^{[l]}$, and scale back.
> Backward, $dA^{[l]}$ multiply with $D^{[l]}$, and scale back.
# Dropout during forward propogation
> Here $D^{[l]}$ would have the same dimension as $A^{[l]}$. And it serves as a mask, which is a matrix of 0 and 1. 

> AFTER GAINING $A^{[l]}$, multiply it with $D^{[l]}$ element-wise.
> One thing good about this is that true is 1 and false is 0 in python. So we can use the mask directly to multiply the activation matrix., representing shutting down.

> Remember to divide by the keep_prob to make sure the expected value of the activation is the same as before.

> For layers have more neurons, we could set the probability of keeping the neuron to be lower to shut down more neurons.

```python
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])                                        
    D1 = D1 < keep_prob                                         
    A1 = np.multiply(A1, D1)                                         
    A1 /= keep_prob                                
    
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])       
    D2 = D2 < keep_prob                        
    A2 = np.multiply(A2, D2)                                       
    A2 /= keep_prob                                         
   
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
```

# Dropout during back propogation
> We need to shutdown the same neurons as we did during the forward propogation. So we need to cache the same mask D1 and D2 to shut down the same neurons.

> AFTER!!!!! gaining $dA^{[l]}$ as usual, we need to apply the same mask $D^{[l]}$ to $dA^{[l]}$ shut down the same neurons. 

> ITS $D^{[l]}$ to $dA^{[l]}$, not $A^{[l]}$!!!

code:
```python
dAl = np.multiply(dAl, Dl)           
dAl /= keep_prob # Dont forget to scale it back

```
```python
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)
    
    # After we gain dA2 normally, we need to apply the same mask D2 to shut down the same neurons.
    dA2 = np.multiply(dA2, D2)           
    dA2 /= keep_prob # Dont forget to scale it back


    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    dA1 = np.dot(W2.T, dZ2)

    dA1 = np.multiply(dA1, D1)           
    dA1 /= keep_prob     

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    return gradients
```