For gradient checking, one thing is that it is often expensive to calculate so we tend to check from time to time rather than the whole iteration procedure.

Idea is: nudge our parameters a little bit, put them into forward propagation and see if the cost function changes accordingly. And ultimately, whether the change is consistent with the gradient that we calculated.

Here is the basic code template for checking a one dimensional theta:

```python
def gradient_check(x, theta, epsilon = 1e-7):
    thetaplus = theta + epsilon
    thetaminus = theta - epsilon
    J_plus = forward_propagation(x, thetaplus) # use thetaplus to calculate J_plus
    J_minus = forward_propagation(x, thetaminus) # use thetaminus to calculate J_minus
    gradapprox = (J_plus - J_minus) / (2. * epsilon)
  
    grad = backward_propagation(x, theta)
    
    numerator = np.linalg.norm(grad - gradapprox) 
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator
```
> Norm is the calculation of L2 Norm, which is essentially length of the vector