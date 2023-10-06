## Padding
np.pad will do
## Single Convolution
```python
def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev, W) # element-wise multiplication
    Z = np.sum(s)
    Z = Z + b
    return Z
```

## Forward pass
**Reminder**:
The formulas relating the output shape of the convolution to the input shape is:
$$ n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
$$ n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1 $$
$$ n_C = \text{number of filters used in the convolution}$$
```python
# GRADED FUNCTION: conv_forward

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """ 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape # m: number of examples
    (f, f, n_C_prev, n_C) = W.shape    # n_C: how many filters
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))     
    A_prev_pad = zero_pad(A_prev, pad)      # Padding 
    
    for i in range(m):              
        a_prev_pad = A_prev_pad[i]      # Select ith training example's after padding        
        for h in range(n_H):                         
            for w in range(n_W):                      
                for c in range(n_C):                   
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f        
                
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])  # c-th filter's W，b
    
   
    cache = (A_prev, W, b, hparameters) 
    return Z, cache
```