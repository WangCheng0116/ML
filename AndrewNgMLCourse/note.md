# Table of Contents

- [Linear Regression with one variable](#linear-regression-with-one-variable)
- [Matrix Review](#matrix-review)
- [Linear Regression with multiple variables](#linear-regression-with-multiple-variables)
- [Features and Polynomial Regression](#Features-and-Polynomial-Regression)
- [Logistic Regression](#Logistic-Regression)


# Linear Regression with one variable
Our hypothesis model:  
![image](https://github.com/WangCheng0116/ML/assets/111694270/c71fb25e-86cb-4792-a6f0-41cd991fe4b1)  
the function to estimate our errors between real data and our model  
![image](https://github.com/WangCheng0116/ML/assets/111694270/893c1a2c-55de-4404-9524-09e58f1a80ea)  
This is the diagram of lost function in a 3d space  
<img width="181" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/9ddb83aa-4fdf-4b60-9baa-3c50ee5aec8d">  
## Gradient Descent  
The idea behind gradient descent is as follows: Initially, we randomly select a combination of parameters (θ₀, θ₁, ..., θₙ), calculate the cost function, and then look for the next combination of parameters that will cause the cost function to decrease the most. We continue to do this until we reach a local minimum because we haven't tried all possible parameter combinations, so we cannot be sure if the local minimum we find is the global minimum. Choosing different initial parameter combinations may lead to different local minima.

<img width="290" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/bcb4b061-b16e-4aac-a465-465d44927113">  
<img width="347" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/2503e93d-41b1-441d-ab4a-aab878123e73">  

In gradient descent, as we approach a local minimum, the algorithm automatically takes smaller steps. This is because when we get close to a local minimum, it's evident that the derivative (gradient) equals zero at that point. Consequently, as we approach the local minimum, the gradient values naturally become smaller, leading the gradient descent to automatically adopt smaller step sizes. This is the essence of how gradient descent works. Therefore, there's actually no need for further decreasing the learning rate (alpha).  

## Apply Gradient Descent into linear regression model  
the key here is to find the partial derivative of model parameter  
![image](https://github.com/WangCheng0116/ML/assets/111694270/0ed8abe7-9d4c-4f81-81e3-cfcb3582f83f)
![image](https://github.com/WangCheng0116/ML/assets/111694270/8c09e8a8-e987-4034-8d63-d7e81979101f)  
this method would sometimes also be referred as "Batch Gradient Descent", because each step makes use of all training datas 

# Matrix Review  
[CS2109S CheatSheet](https://coursemology3.s3.ap-southeast-1.amazonaws.com/uploads/attachments/e3/2d/be/e32dbead0b0c575ef71d120059c1741af17a3085cba0fb5eb6278da9568d8289.pdf?response-content-disposition=inline%3B%20filename%3D%22matrix_calculus_cheatsheet.pdf%22&X-Amz-Expires=600&X-Amz-Date=20230928T083209Z&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2EQN7A45RM2X7VPX%2F20230928%2Fap-southeast-1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Signature=a2bdde670ac0db2119bc0d20f73ffbaa1253b613d99f5c6ec52ee301b0fd9c29)

# Linear Regression with multiple variables  
<img width="237" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/c0b0e60b-18cd-4f48-b66b-83bffe639546">  

## New notations  
$x^{(i)}$ means ith data from the training set, for example,  
![image](https://github.com/WangCheng0116/ML/assets/111694270/4a904451-8f94-4bda-8923-8798c0057112)  
is the second row in the training set  
$x_{j}^{i}$ means the jth feature in the ith data row  
Suppose we have data $x_{1}$, $x_{2}$, $x_{3}$, $x_{4}$, we would add a bias column $x_{0}$ to be 1, so the feature matrix would have a dimension of (m, n + 1)

## Gradient Descent for Multiple Variables
$h_θ (x)=θ^T X=θ_0+θ_1 x_1+θ_2 x_2+...+θ_n x_n$  
taking derivatives, we have  
<img width="400" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/3c084c79-538f-4103-b964-bafb5865e039">  

## Python Implementation  
``` python
def cost_function(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    squared_errors = (predictions - y) ** 2
    J = 1 / (2 * m) * np.sum(squared_errors)
    return J

# Define the gradient descent function
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    J_history = []

    for _ in range(num_iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta -= learning_rate * gradient
        J_history.append(cost_function(X, y, theta))

    return theta, J_history
```

## Feature Scaling  
$x_n=(x_n-μ_n)/s_n$ would lead to a rounder contour hence resulting in less iterations.  

## Learning Rate  
small lr -> slow convergence  
large lr -> J(θ) may not decrease at each iteration; may not converge  


# Features and Polynomial Regression  
Define our own feature:  
instead of $h_θ (x)=θ_0+θ_1×frontage+θ_2×depth$  
we can have $h_θ (x)=θ_0+θ_1×frontage+θ_2×depth$ where $x=frontage*depth=area$  

Polynomial Model:  
$h_θ (x)=θ_0+θ_1 (size)+θ_2 (size)^2$, and then we treat it as a normal linear regression model.  
* Feature Scaling is often needed to avoid huge numbers.

  
# Normal Equation  
My own interpretation: Instead of thinking of it from calculus perspective, let's do it in matrix:  
We want to find a x_hat where it will lead to minimal error, based on this property, we will know the only possible solution lies in the project of vector y onto column space of X, so we have 
$X^T (y - Xθ) = 0$  
therefore $θ=(X^T X)^{(-1)} X^T y$  

Rigorous steps:  
![image](https://github.com/WangCheng0116/ML/assets/111694270/e749d497-1c98-4dbf-b308-af4c0598802b)
![image](https://github.com/WangCheng0116/ML/assets/111694270/e66dd445-e43f-43f1-99ff-db47a668b6fa)
<img width="470" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/6e9c4333-23d2-462c-a124-c1a0b2e3d049">  
We know that $dAB/dB=A^T$ and $(dX^T AX)/dX=2AX$, so  
<img width="470" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/0867dcb7-0c53-489d-a53d-181ceb701b09">  
let it equal to 0.





# Comparison  
| Feature               | Gradient Descent                   | Normal Equation                 |
|-----------------------|-----------------------------------|----------------------------------|
| Learning Rate α       | Required                          | Not Required                     |
| Iterations Required   | Required                          | Computed Once                    |
| For large sample size | Good                              | Troublesome to calculate inverse , which will cost $O(n^{3})$. if n < 10000 would be acceptable |
|Applicabilty           | Various Models                    | Only for linear models            |

## What if X^T X is not invertible?
Reason: 1. redundant features causing it to be linearly dependent  
2. m < n (sample size is less than the number of features) => delete features or use regularization  

# Logistic Regression

## Classification Problems
Yes or No => y ∈ {0 , 1}  
Our hypothesis is $h_θ (x)=g(θ^T X)$, where g is sigmoid function, $g(z)=1/(1+e^{(-z)})$











