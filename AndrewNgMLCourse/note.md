# Table of Content
- [Linear Regression with one variable](#linear-regression-with-one-variable)
- [Matrix Review](#matrix-review)
- [Linear Regression with multiple variables](#linear-regression-with-multiple-variables)
- [Features and Polynomial Regression](#Features-and-Polynomial-Regression)
- [Logistic Regression](#Logistic-Regression)
- [Regularization](#Regularization)
- [Neural Networks](#Neural-Networks)
- [Neural Networks: Learning](#neural-networks-learning)


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
this method would sometimes also be referred to as "Batch Gradient Descent" because each step makes use of all training data 

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
<img src="https://github.com/WangCheng0116/ML/assets/111694270/b1a509ae-2239-4082-a4ca-4f515f258a8e" alt="Image" width="400" height="500">







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
The interpretation of our hypothesis could be $h_θ (x)$ estimates the probability that y = 1 on input x  

## Decision Boundary  
Suppose when $h_θ (x)$ ≥ 0, we predict y = 1, it is equivalent to say $θ^T X$ ≥ 0 based on graph of sigmoid function  
For a linear function, let's assume the boundary is $-3+x_1+x_2≥0$  
<img width="150" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/fc995d40-9379-49df-af50-1e5ad89d2cd9">  
For non-linear, we could also include polynomials like this  $h_θ (x)=g(θ_0+θ_1 x_1+θ_2 x_2+θ_3 x_1^2+θ_4 x_2^2 )$  
<img width="181" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/6422cc39-2385-4b9d-87f7-f72a38cc0961">  

## Cost Function
If we still plug our function into Square Error function, the ultimate cost function would not have the property of convexity, so we need to come up with a new convex function to make sure that local min is global min.  

<img width="400" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/fbf616b0-46f5-45ac-964c-3ea93ed2d81f">
<img width="400" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/63734c4a-9d64-41aa-a771-8ecf3fea90e1">  

Simplify it as $Cost(h_θ (x),y)=-y×log(h_θ (x))-(1-y)×log(1-h_θ (x))$  
Our ultimate cost function is ![image](https://github.com/WangCheng0116/ML/assets/111694270/ecdad8f3-bcb9-4eb4-8b69-3978d91f93b1)  
``` python
# code for cost function
def cost(theta, X, y):
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
```
Taking derivative, we found out that 
<img src="https://github.com/WangCheng0116/ML/assets/111694270/deebcf4b-ada9-4ca3-8b34-14f4e2bcb9a8" alt="Image" width="500" height="450">
 
it's the same as linear regression!!!  
we apply the same strategy here  
<img width="400" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/01ef33e8-275d-4025-bf57-c3c000001273">  
in this case feature scaling is also helpful in reducing the number of iterations

## Multiclass Classification  
<img width="400" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/07a1202e-ee1d-4a2c-968b-b3a791486907">  

We would use a method called one v.s. all  
<img width="400" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/e42adf14-d933-42b1-89c0-229927c8a238">  

in this case we have three classifiers $h_θ^{(i)} (x)$. In order to make predictions, we will plug in x to each classifier and select the one with the highest probability $max h_θ^{(i)} (x)$

# Regularization  
## Underfit and Overfit
<img width="420" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/135b056f-de7e-439d-a66b-3f07f111e5ac">  

* Underfit: High bias
* Just Right
* Overfit: High variance | We have a very small error but this model is not generalizable

Solutions:
* Reduce the number of features
  * manually select
  * model selection algorithm
* Regularization
  * keep all features but reduce magnitude of parameters

## Motivation to have regularization
$h_θ (x)=θ_0+θ_1 x_1+θ_2 x_2^2+θ_3 x_3^3+θ_4 x_4^4$, the reason why we have overfit is because of those terms with high degree, we want to minimize their impact on our model. So we choose![image](https://github.com/WangCheng0116/ML/assets/111694270/ada44f0f-d354-4b75-a042-afe7719eaeb1)  

to include punishment for high degree terms.  

In general, we will have![image](https://github.com/WangCheng0116/ML/assets/111694270/dd0518a4-c7a0-4fb5-845b-ffd8e021824d)  
where λ is called **Regularization Parameter**


## Gradient Descent and Normal Equation Using Regularization in Linear Regression


> important: $θ_0$ would never be part of regularization, so all index numbers of $θ_j$ start from 1

![image](https://github.com/WangCheng0116/ML/assets/111694270/f5b44dfa-13fd-4744-aaf2-14fca072161b)
![image](https://github.com/WangCheng0116/ML/assets/111694270/a4a32b91-7e79-4002-a218-e923f3c22d56)  
* Note that the difference is each time $θ_j$ is multiplied with a number smaller than 1
* For $θ_0$, we take λ as 0

<img width="227" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/e95e8711-5f8e-4f9e-9ff8-ecf101f99232">  

Rigorous proof:  

<img src="https://github.com/WangCheng0116/ML/assets/111694270/756b269a-6734-4f3a-964a-d29ef957b81b" alt="Image" width="550" height="400">


* If λ > 0, it is guaranteed to be invertible;  

* The 0, 1, 1 ... 1 matrix is of size $(n + 1) * (n + 1)$


## Gradient Descent Using Regularization in Logistic Regression
> important: $θ_0$ would never be part of regularization, so all index numbers of $θ_j$ start from 1

![image](https://github.com/WangCheng0116/ML/assets/111694270/38b8dc08-2df9-481f-8a2e-a7492728d1df)
``` python
import numpy as np
def costReg(theta, X, y, lambda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (lambda / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
```
We will have the following update models:  
![image](https://github.com/WangCheng0116/ML/assets/111694270/44736829-0e51-498c-8f1c-69f4e36b0294)  
![image](https://github.com/WangCheng0116/ML/assets/111694270/55b31ab3-1584-4f69-a37a-4161449723b3)

# Neural Networks

## Model Representation I
<img width="700" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/2da8a328-c40f-4342-806c-37784ae657ff">  

Here we can see **Input Layer**, **Hidden Layer** and **Output Layer**, each layer may also have a **Bias Unit**  
$a_i^{(j)}$ means the activation of ith unit in jth layer  
activation means the value computed by it
$Θ^{(j)}$ means the weight matrix from layer j to layer j + 1  
In this case $θ^{(1)}$ is a matrix of size 3 * 4 (take in 4 parameters (including bias term) and outputs three data in layer 3)  
Generally speaking, j layer has m units, j + 1 layer has n units, then $θ^{(j)}$ would be of dimension n x (m + 1)
![image](https://github.com/WangCheng0116/ML/assets/111694270/1477231b-2e14-496e-ba7b-36583c0232b7)
![image](https://github.com/WangCheng0116/ML/assets/111694270/49d8abe0-a051-487b-886b-6340681c6326)
![image](https://github.com/WangCheng0116/ML/assets/111694270/a4a1a214-c641-4b6d-b015-d5d1e3ebdc94)
![image](https://github.com/WangCheng0116/ML/assets/111694270/50d95b4e-c1ed-41f5-947a-daecaa0e7115)

## Model Representation II
Suppose we are at layer j where the data is defined as 
$a^{(j)}$ = $(a_1^{(j)}, a_2^{(j)}, a_3^{(j)})^T$  
we need to first add a bias term 1 to the layer $(a_0^{(j)}, a_1^{(j)}, a_2^{(j)}, a_3^{(j)})^T$,  
then the data after weighting ready to be fed to j + 1 layer is defined as (suppose j+1 layer has 3 units)   
$z^{(j+1)}$ = $(z_1^{(j+1)}, z_2^{(j+1)}, z_3^{(j+1)})^T$  
using the weighted matrix, we have $z^{(j+1)}$ = $Θ^{(j)} a^{(j)}$  
but its just the date propagated to j+1 layer, they haven't been digested,  
after digestion, $a^{(j + 1)} = sigmoid(z^{(j + 1)})$  


## Examples 
<img width="320" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/dc4cfa59-007c-4174-aa77-908b92708efc">  

here is one possible way for us to use neurons to achieve **or** function

## Multiclass Classification
The intuition remains the same, just one thing different. The final output is a set of data rather than a single one, for example, [1 0 0], [0 1 0], [0 0 1] mean cars, cats and dogs respectively.

# Neural Networks: Learning
## Cost Function  
 ![image](https://github.com/WangCheng0116/ML/assets/111694270/8e3c6b71-3b5b-4564-8712-2e0fe61fdefe)  

It looks intimidating, but the idea remains the same, the only difference is that the final output is a vector of dimension **K**, $(h_θ(x))_i$ = i^th output in the result vector, the last summation term is just simply the sum of all the parameters (all weights) excluding the term where i = 0, which means it is multiplied with bias, so there's no need to take them into account. (If you do nothing will happen)  
$θ_{ji}^{(l)}$ here represents a parameter or weight in a neural network, specifically for layer "l" connecting the "i-th" neuron in layer "l" to the "j-th" neuron in layer "l+1" mathematically, $θ_{ji}^{(l)}$ is weight connecting $a_i^{l}$ and $a_j^{l+1}$  
A concrete example: $θ_{10}^{(2)}$ is the weight connecting $a_0^{(2)}$ at layer 2 with $a_1^{(3)}$

## Backpropagation Algorithm
The idea is to compute the error term of each layer and then use it to update the weights.  
Our goal always remain the same: we have a set of parameters waiting to be optimized, and we also have cost function. The essence of gradient descent is to find the partial derivative of cost function to all the parameters.  


Let's start with the notation in the proof:  
For weights let **wᴸₘₙ** be the weight from the mₜₕ neuron in the (L-1)ₜₕ layer to the nₜₕ neuron in the Lₜₕ layer.  
activation of the **mₜₕ** neuron in the Lₜₕ layer by aᴸₘ and for the bias we’ll do **bᴸₘ**.  
So we will have![image](https://github.com/WangCheng0116/ML/assets/111694270/687fb62c-30b3-45e4-8908-6d43d108941b)
![image](https://github.com/WangCheng0116/ML/assets/111694270/8924c514-47f5-4093-98f4-0610be5a4444)
![image](https://github.com/WangCheng0116/ML/assets/111694270/befbe5ce-cfa8-44ff-b5b2-92e9de4c0f90)  
is same as  
![image](https://github.com/WangCheng0116/ML/assets/111694270/8c13ccb7-6e96-411b-9c0c-960a1023bb3a)  
and only when **i=n**, the term would survive in the partial derivative, so we have  
![image](https://github.com/WangCheng0116/ML/assets/111694270/b821315a-17dd-44cb-b9d1-849ba06e7aab)  
Let the red part be **δᴸₙ**, we have  
![image](https://github.com/WangCheng0116/ML/assets/111694270/997698b4-3d3a-4a17-84a2-9ff04c4bd367)  
For bias term, we use the same method:  
![image](https://github.com/WangCheng0116/ML/assets/111694270/a489afdd-9578-4875-8c23-b4a689cfc601)  
plugging in 
![image](https://github.com/WangCheng0116/ML/assets/111694270/52be6ccd-b600-4fb3-b1ff-3d286d8ddb9c)  
we will have  
![formula](https://github.com/WangCheng0116/ML/assets/111694270/2f4506ec-ad95-4e94-820c-3a94c04d3f34)  
our goal here simplifies to find delta  

Let's start with the last layer H  
![image](https://github.com/WangCheng0116/ML/assets/111694270/fd3874c4-582c-4877-be6c-ddc56b4a04f2)  
since we know the cost function, we have  
![image](https://github.com/WangCheng0116/ML/assets/111694270/9f877d79-57d8-4dc1-9b11-9778577008ad)  
this is the first part of the last last equation  
![image](https://github.c. m/WangCheng0116/ML/assets/111694270/7882297c-ae99-4976-b333-ad1422951954) 
this is the second term of the last last equation, by combining both  
![image](https://github.com/WangCheng0116/ML/assets/111694270/97674f42-6254-4369-aff6-622de1a40876)  
By generalizing into the relation between L and L + 1, marked as red here, we still need to find out the second term (notice that only when **i=n** it will be left)   
![image](https://github.com/WangCheng0116/ML/assets/111694270/ad26a84b-6ec5-4988-92a7-aaf67e04aa39)  
![image](https://github.com/WangCheng0116/ML/assets/111694270/cbd9cf1f-c8b8-4c81-be4c-527b151b66e8)  
By combining two terms, the last last last formula would be  
![image](https://github.com/WangCheng0116/ML/assets/111694270/b0d062fa-96bc-4af4-984a-c0ed5b9ae5a6)  
in vectorization form, this can be also written as (⊙ means elementwise production)  
![image](https://github.com/WangCheng0116/ML/assets/111694270/5aa0065a-c3ec-4ea4-a297-d125e8ab2e16)  

To sum up, in two versions, the first is from elementwise perspective  
![image](https://github.com/WangCheng0116/ML/assets/111694270/8db186f9-46a3-4910-b8bc-5ca16ff5aa91)  
the other one is from vector perspective (layer by layer)    
![image](https://github.com/WangCheng0116/ML/assets/111694270/0e67b7a1-445c-484b-a85a-8439e295fdc2)














































