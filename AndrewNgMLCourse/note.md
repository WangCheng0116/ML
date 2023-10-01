# Table of Content
- [Linear Regression with one variable](#linear-regression-with-one-variable)
- [Matrix Review](#matrix-review)
- [Linear Regression with multiple variables](#linear-regression-with-multiple-variables)
- [Features and Polynomial Regression](#Features-and-Polynomial-Regression)
- [Logistic Regression](#Logistic-Regression)
- [Regularization](#Regularization)
- [Neural Networks](#Neural-Networks)
- [Deep Neural Network](#Deep-Neural-Network)
- [Setting Up ML Application](#Setting-Up-ML-Application)
- [Optimization Algorithms](#Optimization-Algorithms)
- [Hyperparameter Tuning Batch Normalization and Programming Frameworks](#Hyperparameter-Tuning-Batch-Normalization-and-Programming-Frameworks)
- [Introduction to ML Strategy I](#Introduction-to-ML-Strategy-I)
- [Introduction to ML Strategy II](#Introduction-to-ML-Strategy-II)



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

## Vectorization of Logistic Regression  
``` python
Z = np.dot(X, w) + bias
A = sigmoid(Z)
error = A - y
dw = np.dot(X.T, error) / m
db = np.sum(error)/ m
w = w - α * dw
bias = b - α * db
```

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

## Gradient Descent Using Regularization in Neural Networks  
<img width="1416" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/40ae8cb4-9d36-4afb-bbe1-c75da556a49e">  
Basically, it means for all $w^{[i]}$, it will sum up the square of each element in $w^{[i]}$, i.e., the sum of the square of all weight parameters. We need to include an extra term when updating.  
<img width="620" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/6f07840e-ddf6-4c28-b1ab-f988811d015e">  

## Dropout  
### Implementation
One way to implement this is called **inverted-dropout**  
``` python
keep_prob = 0.8    # Ses a probability threshold to keep a neuron
dl = np.random.rand(al.shape[0], al.shape[1]) < keep_prob # less than threshold? keep it
al = np.multiply(al, dl) # some neuron will be multiplied with a false (0), hence being eliminated
al /= keep_prob # reverse the multiplication to keep expected average value the same
```
## Other Methods of Regularization

* Data Augmentation

Data Augmentation is a technique used in deep learning to increase the size of the training and validation datasets by applying various transformations to images. These transformations may include flipping, local zooming followed by cropping, and more. The goal is to generate more diverse data points for training and validation, which can help improve the generalization of the model.

* Early Stopping

Early Stopping is a regularization technique employed during the training of neural networks. It involves monitoring the cost (or loss) on both the training and validation datasets during the gradient descent process. The cost curves for both datasets are plotted on the same axis. The training is halted when the training error continues to decrease, but the validation error starts to increase significantly. This is done to prevent overfitting, as a significant gap between the training and validation errors indicates that the model is becoming too specialized in the training data and may not generalize well to unseen data. However, one limitation of this approach is that it cannot simultaneously achieve the optimal balance between bias and variance, as it primarily focuses on avoiding overfitting.

Using a combination of *Data Augmentation* and *Early Stopping* can help in training deep learning models that are better at generalizing to new, unseen data while avoiding overfitting.




# Neural Networks

## Model Representation I
<img width="1131" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/e7a58359-6a69-4131-a5b5-62b14556ea6f">  

**z** is pre-activation value, **a** is after activation, $w^{[i]}$ means the weight matrix from layer **i-1** to to layer **i**, the and size of matrix $w^{[i]}$ is (# of nodes in layer **i**, # of nodes in layer **i-1**)  
In vectorization form, we will have  
<img width="688" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/f42be892-8d4c-46e6-bb1a-1a0db0c045fe">  
<img width="390" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/d354d725-3558-4169-9a07-af35e0ede829">  
After vectorization, we will only need to change x to X and a to A,  
<img width="920" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/205eff58-ff7e-476c-b6d7-7c015d84e83f">

## Other activation functions  

* **tanh (Hyperbolic Tangent):** 
  * Formula: $tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$
  * Derivative: $tanh'(z) = 1 - tanh^2(z)$
  * Explanation: The tanh function maps its input to values in the range [-1, 1]. Like the sigmoid function, it's also used in neural networks, particularly in the hidden layers. One problem with tanh and sigmoid is the vanishing gradient problem, which can occur when values are too small or too large, leading to slow convergence during training.

* **ReLU (Rectified Linear Unit):** 
  * Formula: $ReLU(z) = \max(0, z)$
  * Derivative: $ReLU'(z) = 1$ if $z > 0$, $ReLU'(z) = 0$ if $z \leq 0$
  * Explanation: ReLU is a widely used activation function that replaces negative values with zero and leaves positive values unchanged. It helps mitigate the vanishing gradient problem and accelerates training. It's commonly used in the hidden layers of neural networks. For binary classification, sigmoid is often used in the output layer in combination with ReLU in the hidden layers.

* **Sigmoid (Logistic):** 
  * Formula: $Sigmoid(z) = \frac{1}{1 + e^{-z}}$
  * Derivative: $Sigmoid'(z) = Sigmoid(z) \cdot (1 - Sigmoid(z))$
  * Explanation: The sigmoid function maps its input to values in the range (0, 1). While it was commonly used in the past, especially in the output layer for binary classification, it has some drawbacks, including the vanishing gradient problem. It's not used as frequently as ReLU in modern deep learning architectures but can still be useful in certain cases.

## Multiclass Classification
The intuition remains the same, just one thing different. The final output is a set of data rather than a single one, for example, [1 0 0], [0 1 0], [0 0 1] mean cars, cats and dogs respectively.

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
In vectorization form, we have  
![image](https://github.com/WangCheng0116/ML/assets/111694270/35b77c72-8cda-40bc-8bb6-351679756089)  

For bias terms, we use the same method:  
![image](https://github.com/WangCheng0116/ML/assets/111694270/a489afdd-9578-4875-8c23-b4a689cfc601)  
plugging in 
![image](https://github.com/WangCheng0116/ML/assets/111694270/52be6ccd-b600-4fb3-b1ff-3d286d8ddb9c)  
we will have  
![formula](https://github.com/WangCheng0116/ML/assets/111694270/2f4506ec-ad95-4e94-820c-3a94c04d3f34)  
in vectorization form  
![image](https://github.com/WangCheng0116/ML/assets/111694270/24eaf15e-c671-4430-a3c1-5b35c9cfc8b6)

our goal here simplifies to find delta  

Let's start with the last layer H  
![image](https://github.com/WangCheng0116/ML/assets/111694270/fd3874c4-582c-4877-be6c-ddc56b4a04f2)  
since we know the cost function, we have  
![image](https://github.com/WangCheng0116/ML/assets/111694270/9f877d79-57d8-4dc1-9b11-9778577008ad)  
this is the first part of the last last equation  
![image](https://github.com/WangCheng0116/ML/assets/111694270/7882297c-ae99-4976-b333-ad1422951954) 
this is the second term of the last last equation, by combining both  
![image](https://github.com/WangCheng0116/ML/assets/111694270/97674f42-6254-4369-aff6-622de1a40876)  
in vectorization form  
![image](https://github.com/WangCheng0116/ML/assets/111694270/9fb3cae8-5ad1-419a-94eb-8531463610d3)  

By generalizing into the relation between L and L + 1, marked as red here,     
![image](https://github.com/WangCheng0116/ML/assets/111694270/9ca6e8ad-d8bb-4c7f-a931-835a7d19f603)  
we still need to find out the second term (notice that only when **i=n** it will be left),  
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

The final procedure would be  
1- Find aᴸ and zᴸ for layers 0 through H by feeding an example into the network. (Use the first two equations.) This is known as the “forward pass”.

2- Compute δᴸ for layers H through 0 using the formulae for δᴴ, δᴸ respectively. (Use the third equation.)

3- Simultaneously compute ∂J/∂Wᴸ and ∂J/∂bᴸ for layers H through 0 as once we have δᴸ we can find both of these. (Use the last two equations.) This is known as the “backward pass”.

4- Repeat for more examples until the weights and biases of the network can be updated through gradient descent (depends on your batch size):

<img width="801" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/b21ef5e0-3b1b-4780-8e2a-bd5069012506">

## Randomize Initialization  
If we still use 0 to initialize, it is easy to prove the whole updating procedure will be completely symmetric.  
so what we will do is initialize $w^{[i]}$ with a random number from [0, 1] * 0.01, the reason is we want it to have values close to 0 where the slope will be relatively large.  
for $b^{[i]}$, it won't affect symmetry so we could initialize it to be 0


# Deep Neural Network
## Working Flow Chart  
![image](https://github.com/WangCheng0116/ML/assets/111694270/b7a21d78-fbeb-40e3-bc89-61f6dbaf5c68)  
By caching the value of $z^{[i]}$, we could make bp easier.  
<img width="1313" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/300facb2-a4df-463e-b31d-4c8923edea1b">
<img width="1416" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/7b569246-bebd-407f-b50f-66745f2ae7e8"><img width="1416" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/a1d3c3bc-fcba-42de-9abe-d74bb9ee3570">  
in these two detailed slides, LHS refers to single training data, while the other one is after vectorization.  

# Setting Up ML Application  

## Data Split: Training / Validation / Test Sets

Applying deep learning is a typical iterative process.

In the process of building a model for a problem with sample data, the data is divided into the following parts:

* **Training set**: Used to **train** the algorithm or model.

* **Validation set (development set)**: Utilized for **cross-validation** to **select the best model**.

* **Test set**: Finally used to test the model and obtain an **unbiased estimate** of the model's performance.

In the **era of small data**, with dataset sizes like 100, 1000, or 10000, data can be split according to the following ratios:

* Without a validation set: 70% / 30%.

* With a validation set (also known as simple cross-validation set): 60% / 20% / 20%.

However, in today's **big data era**, datasets for a problem can be on the scale of millions. Therefore, the proportion of the validation and test sets tends to become smaller.


## Vanishing and Exploding Gradients  
[A pretty nice introduction](https://zhuanlan.zhihu.com/p/72589432), sigmoid actually sucks because its derivative is always less than 0.25, very likely to cause Vanishing Gradients. On the other hand, the derivative of ReLU always 1 when x > 0, NIICE!!  


How can we solve the issue?  

*  Xavier initialization
In the Xavier Initialization, the goal is to keep the values of weights $w_i$ relatively small when the number of inputs (n) is large. This helps ensure that the weighted sum (z) remains reasonably small.

To achieve smaller weights $w_i$, the variance of each weight is set to 1/n. Here's the formula for Xavier Initialization:

```python
W = np.random.randn(W.shape[0], W.shape[1]) * np.sqrt(1/n)
```

* KaiMing He Initialization
The only difference is Var(wi)=2/n, suitable when choosing ReLU.

## Gradient Checking (Grad Check)  
take $W^{[1]}$，$b^{[1]}$，...，$W^{[L]}$，$b^{[L]}$ them all out and form a giant vector θ, like  

$$J(W^{[1]}, b^{[1]}, ..., W^{[L]}，b^{[L]}) = J(\theta)$$

Same technique applies to $dW^{[1]}$，$db^{[1]}$，...，$dW^{[L]}$，$db^{[L]}$ to form dθ,  

$$d\theta_{approx}[i] ＝ \frac{J(θ_{1}, θ_{2}, ..., θ_i+\varepsilon, ...) - J(θ_1, θ_2, ..., θ_i-\varepsilon, ...)}{2\varepsilon}$$

it should satisfy

$$\approx{d\theta[i]} = \frac{\partial J}{\partial \theta_{i}}$$

so we could calculate the difference between them, and if the difference is small enough, we could say our gradient is correct. (denominator is for scaling)

$$\frac{{||dθ_{approx} - d\theta||}_2}{{||dθ_{approx}||}_2+{||d\theta||}_2}$$

where

$${||x||}_2 = \sum^N_{i=1}{|x_i|}^2$$  

* if it is smaller than $10^{-7}$, we could say our gradient is correct.  
* if it is between $10^{-7}$ and $10^{-4}$, we could say it is suspicious.  
* if it is larger than $10^{-4}$, we could say our gradient is probably wrong.


# Optimization Algorithms

## Mini-Batch Gradient Descent
if we process a subset of the training data each time, using gradient descent, our algorithm will execute faster. These small subsets of training data that are processed at a time are called **mini-batches**.
And the algorithm is called **Mini-Batch Gradient Descent**.

If the mini-batch size is 1, then it is called **Stochastic Gradient Descent**.
If the mini-batch size is m, then it is called **Batch Gradient Descent**.

Here is a picture to illustrate the difference between the three algorithms:
<img width="834" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/5d97c96a-ba73-4dcf-9286-662e6cd7a015">
<img width="834" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/d30945a7-05a6-4e09-b6dd-3e28ac72fef5">


### Batch Gradient Descent:
- Performs one gradient descent update using all m training samples.
- Each iteration takes a long time, making the training process slow.
- Relatively lower noise but larger magnitude.
- The cost function always decreases in the decreasing direction.

### Stochastic Gradient Descent:
- Performs one gradient descent update for each training sample.
- Training is faster but loses the computational acceleration from vectorization. (Since we have to process each sample one by one)
- Contains a lot of noise; reducing the learning rate can be appropriate.
- The overall trend of the cost function moves closer to the global minimum but never converges and keeps fluctuating around the minimum.

Therefore, choosing an appropriate size `1 < size < m` for Mini-Batch Gradient Descent enables fast learning while also benefiting from vectorization. The cost function's descent lies between the characteristics of Batch and Stochastic Gradient Descent. Hence, we can have two benifits at the same time:
- By using vectorization, we can speed up the training process.
- We don't have to use the whole training set, which will also reduce the time.

## Exponentially Weighted Averages

$$
v_t = 
\begin{cases} 
\theta_1, &t = 1 \\\\ 
\beta v_{t-1} + (1-\beta)\theta_t, &t > 1 
\end{cases}
$$
where $\beta$ is the weight, $S_t$ is the exponentially weighted average (from day $1$ till day $t$), $Y_t$ is the current true value.

Suppose $\beta = 0.9$, 
![Alt text](image.png)  

> The essence is an exponentially weighted moving average. Values are weighted and exponentially decay over time, with more weight given to recent data, but older data is also assigned some weight.

We can see that the process of calculating the exponential weighted average is actually a recursive process, which has a significant advantage. When I need to calculate the average from $1$ to a certain moment $n$, I don't need to keep all the values at each moment, sum them up, and then divide by n, as in a typical average calculation.

Instead, I only need to retain the average value from $1$ to $n-1$ moments and the value at the nth moment. In other words, I only need to keep a constant number of values and perform calculations. This is a good practice for reducing memory and space requirements, especially when dealing with massive data in deep learning.

![Alt text](image-1.png)  
A larger β corresponds to considering a greater number of days in calculating the average, which naturally results in a smoother and more lagged curve.  
yellow: β = 0.5
red: β = 0.9
green: β = 0.98

``` python
v = 0
for t in range(1, len(Y)):
  v = beta * v + (1 - beta) * Y[t]
```

## Bias Correction in Exponentially Weighted Averages
The problem with the exponentially weighted average is that it starts from 0, so the first few values are not very accurate. They tend to be very smaller.

To solve this problem, we can use bias correction. The idea is to divide the value by $1 - \beta^t$ to make the average value more accurate.

The math formula is as follows:
$$v_t = \frac{\beta v_{t-1} + (1 - \beta)\theta_t}{{1-\beta^t}}$$

One thing good about this is that when t is large, the denominator will be close to 1, so it won't affect the value too much. But when t is small, the denominator will be a large number, so it will amplify the value.

## Gradient Descent with Momentum
$$v_{dW} = \beta v_{dW} + (1 - \beta) dW$$
$$v_{db} = \beta v_{db} + (1 - \beta) db$$
$$W := W - \alpha * v_{dW}$$
$$b := b - \alpha * v_{db}$$
initial value of $v_{dW}$ and $v_{db}$ is 0  
$\alpha$ and $\beta$ are hyperparameters here, usually $\beta$ is set to 0.9

## RMSprop(Root Mean Square Propagation)

$$s_{dw} = \beta s_{dw} + (1 - \beta)(dw)^2$$
$$s_{db} = \beta s_{db} + (1 - \beta)(db)^2$$
$$w := w - \alpha \frac{dw}{\sqrt{s_{dw} + \epsilon}}$$
$$b := b - \alpha \frac{db}{\sqrt{s_{db} + \epsilon}}$$
$\epsilon$ is a small number to avoid division by zero, usually set to $10^{-8}$, and square and square root are element-wise operations.

The intuition is that if the derivative ($s_{dw}$ and $s_{db}$) is large, the denominator in $w$ or $b$ will be small, so the update will be small. If the derivative is small, the denominator will be large, so the update will be large. This will make the update more stable.

In English: If ont direction of gradient is pretty large, it will take small steps in this direction. And vice versa.
![Alt text](image-2.png)

## Adam Optimization Algorithm(Adaptive Moment Estimation)
The Adam algorithm combines the ideas of Momentum and RMSprop. It is the most commonly used optimization algorithm in deep learning.  

This is how it works, in the first iteration, initialize:
$$v_{dW} = 0, s_{dW} = 0, v_{db} = 0, s_{db} = 0$$

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

$\alpha$ is the learning rate, $\beta_1$ and $\beta_2$ are hyperparameters, usually set to 0.9 and 0.999 respectively.

## Learning Rate Decay
If you set a fixed learning rate α, near the minimum point, due to the presence of some noise in different batches, the convergence won't be precise. Instead, it will always fluctuate within a relatively large range around the minimum.

However, if you gradually decrease the learning rate α over time, in the early stages when α is large, the descent steps are significant, allowing for faster gradient descent. As time progresses, α is gradually reduced, decreasing the step size, which helps with algorithm convergence and getting closer to the optimal solution.

The most commonly used learning rate decay methods:

- $$\alpha = \frac{1}{1 + decay\\\_rate * epoch\_num} * \alpha_0$$

- $$\alpha = 0.95^{epoch\\\_num} * \alpha_0$$
- $$\alpha = \frac{k}{\sqrt{epoch\\\_num}} * \alpha_0$$

## Local Optima

- When training large neural networks with a large number of parameters, and the cost function is defined in a high-dimensional space, **getting stuck in a bad local optimum is unlikely**. Because in high dimensional space, it a point is a local optimum, it needs to increse or decrease in all directions, which is very unlikely to happen.
- The presence of plateaus near saddle points can result in very slow learning. This is why momentum gradient descent, RMSProp, and Adam optimization algorithms can accelerate learning. They help to escape from plateaus early.


# Hyperparameter Tuning Batch Normalization and Programming Frameworks

## Tuning Process
**Most Important**:
- Learning rate (α)

**Next in Importance**:
- β: Momentum decay parameter, often set to 0.9
- #hidden units: Number of neurons in each hidden layer
- Mini-batch size

**Less Important, but Still Significant**:
- β1, β2, ϵ: Hyperparameters for the Adam optimization algorithm, commonly set to 0.9, 0.999, and $10^{-8}$ respectively
- #layers: Number of layers in the neural network
- decay_rate: Learning rate decay rate

## Using an Appropriate Scale to Pick Hyperparameters

When we are trying to find the best hyperparameters, we can't just randomly pick a value. Instead, we should use an appropriate scale to pick the value.

For example, for learning rate α, we can't just randomly pick a value between 0 and 1. Instead, we can use a scale like 0.1, 0.01, 0.001, 0.0001, etc. to pick the value.
``` python
r = -4 * np.random.rand() - 1# [-5, -1]
alpha = 10 ** r # [0.00001, 1]
```
Same for β, let's say we want to pick value between 0.9 and 0.999, what we can do is take 1 - β, which will fall into the range of 0.001 to 0.1, then we can use the same scale to pick the value.
``` python
temp = -3 * np.random.rand() # [-4, -1]
beta = 1 - 10 ** temp # [0.9, 0.9999]
```

## Hyperparameters Tuning in Practice: Pandas vs. Caviar

**Panda Approach**:
babysitting one model, and tuning the hyperparameters to get the best performance.

**Caviar Approach**:
train many models in parallel, and choose the one that works best.

## Normalizing Activations in a Network
## Batch Normalization (BN)

**Batch Normalization (often abbreviated as BN)** makes parameter search problems easier, stabilizes the neural network's selection of hyperparameters, expands the range of hyperparameters, works effectively, and makes training easier.

Previously, we applied standardization to input features X. Similarly, we can process the activation values $a^{[l]}$ of **hidden layers** to accelerate the training of $W^{[l+1]}$ and $b^{[l+1]}$. In **practice**, it is common to normalize $Z^{[l]}$:

$$\mu = \frac{1}{m} \sum_i z^{(i)}$$
$$\sigma^2 = \frac{1}{m} \sum_i {(z_i - \mu)}^2$$
$$z_{norm}^{(i)} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Here, m is the number of samples in a single mini-batch, and ϵ is added to prevent division by zero, typically set to $10^{-8}$.

This way, we ensure that all input $z^{(i)}$ have a mean of 0 and a variance of 1. However, we don't want hidden units to always have a mean of 0 and variance of 1. It might be more meaningful for hidden units to have different distributions. Therefore, we calculate:

$$\tilde z^{(i)} = \gamma z_{norm}^{(i)} + \beta$$

Here, γ and β are both learning parameters of the model. They can be updated using various gradient descent algorithms, just like updating the weights of the neural network.

By appropriately setting γ and β, we can set the mean and variance of $\tilde z^{(i)}$ to any desired values. This way, we normalize the hidden layer's $z^{(i)}$ and use the obtained $\tilde z^{(i)}$ instead of $z^{(i)}$.

The reason for **setting γ and β** is that if the mean values of the inputs to each hidden layer are close to 0, i.e., in the linear region of the activation function, it is not conducive to training a nonlinear neural network, resulting in a poorer model. Therefore, we need to further process the normalized results using γ and β.

## Fitting Batch Norm into a Neural Network
![Alt text](image-4.png)
## Batch Normalization and Its Usage

In practice, **Batch Normalization** is often used on mini-batches, which is why it gets its name.

When using Batch Normalization, because the normalization process involves subtracting the mean, the bias term (b) essentially becomes ineffective, and its numerical effect is achieved through β. Therefore, in Batch Normalization, you can omit b or temporarily set it to 0.

During the use of gradient descent algorithms, iterative updates are applied to $W^{[l]}$, $β^{[l]}$, and $γ^{[l]}$. In addition to traditional gradient descent algorithms, you can also use previously learned optimization algorithms like Momentum Gradient Descent, RMSProp, or Adam.

Batch Normalization plays a crucial role in stabilizing and accelerating the training of deep neural networks.

## Why Does Batch Norm Work?
## Benefits of Batch Normalization

Batch Normalization offers several advantages in neural network training:

- **Flattening Gradients**: It helps in making the gradients smoother, preventing the vanishing gradient problem.

- **Optimizing Activation Functions**: Batch Normalization optimizes the activation functions within the network, aiding faster and more effective training.

- **Addressing Gradient Vanishing**: It helps in mitigating the gradient vanishing problem, allowing for training of deep networks.

- **Regularization Effect**: Batch Normalization acts as a form of regularization, promoting better model generalization. Reason: This adds some noise to the values $Z^{[l]}$ within that minibatch. So similar to dropout, it adds some noise to each hidden layer's activations.

## Batch Norm at Test Time

In theory, we could incorporate all training data into the final neural network model and directly use the computed $μ^{[l]}$ and $σ^{2[l]}$ for each hidden layer in the testing process. However, in practical applications, this method is not commonly used. Instead, we apply the previously learned exponential weighted average method to predict the μ and $σ^2$ for individual samples during the testing process.

For the l-th hidden layer, we consider all mini-batches under that hidden layer, and then predict the current individual sample's $μ^{[l]}$ and $σ^{2[l]}$ using exponential weighted averaging. This approach allows us to estimate the mean and variance for individual samples during the testing process.

## Softmax Regression
For **multiclass problems**, let's denote the number of classes as C. In the neural network's output layer, which is the L-th layer, the number of units $n^{[L]} = C$. Each neuron's output corresponds to the probability of belonging to a specific class, i.e., $P(y = c|x), c = 0, 1, .., C-1$. There is a generalized form of logistic regression called **Softmax Regression**, which is used to handle multiclass classification problems.

Softmax Regression is particularly suitable for solving multiclass classification tasks.

A pretty straightforward example:  
![Alt text](image-6.png)

# Introduction to ML Strategy I

## Orthogonalization in Machine Learning

The core principle of **orthogonalization** is that each adjustment only affects one aspect of the model's performance without impacting other functionalities. This approach aids in faster and more effective debugging and optimization of machine learning models.

In a machine learning (supervised learning) system, you can distinguish four "functionalities":

1. Building a model that performs well on the training set.
2. Building a model that performs well on the validation set.
3. Building a model that performs well on the test set.
4. Building a model that performs well in practical applications.

Among them:

- For the first point, if the model performs poorly on the training set, you can try training a larger neural network or switching to a better optimization algorithm (such as Adam).
  
- For the second point, if the model performs poorly on the validation set, you can apply regularization techniques or introduce more training data.

- For the third point, if the model performs poorly on the test set, you can try using a larger validation set for validation.

- For the fourth point, if the model performs poorly in practical applications, it might be due to incorrect test set configuration or incorrect cost function evaluation metrics, which require adjustments to the test set or cost function.

Orthogonalization helps us address various issues more precisely and effectively as we encounter them.

## Training / Validation / Test Set Split

In machine learning, we typically divide the dataset into training, validation, and test sets. When constructing a machine learning system, we employ various learning methods to train different models on the **training set**. Subsequently, we evaluate the quality of these models using the **validation set** and only select a model when we are confident in its performance. Finally, we use the **test set** to perform the ultimate testing of the chosen model.

Therefore, the allocation of training, validation, and test sets is crucial in machine learning model development. Properly setting these datasets can significantly improve training efficiency and model quality.

## Distribution of Validation and Test Sets

The data sources for the validation and test sets should be the same (coming from the same distribution) and consistent with the data the machine learning system will encounter in real-world applications. They must also be randomly sampled from all available data to ensure that the system remains as unbiased as possible.

## Partitioning Data

In the past, when the data volume was relatively small (less than 10,000 examples), data sets were typically divided according to the following ratios:

* Without a validation set: 70% / 30%;
* With a validation set: 60% / 20% / 20%;

This was done to ensure that both the validation and test sets had an adequate amount of data. In the current era of machine learning, data sets are generally much larger, for example, with one million examples. In such cases, setting the corresponding ratios to 98% / 1% / 1% or 99% / 1% is sufficient to ensure that the validation and test sets have a substantial amount of data.

## Comparing to Human Performance

The birth of many machine learning models is aimed at replacing human work, so their performance is often compared to human performance levels.

![Bayes-Error](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Structuring_Machine_Learning_Projects/Bayes-Optimal-Error.png)

The graph above shows the evolution of machine learning systems' performance compared to human performance over time. Generally, when machine learning surpasses human performance, its rate of improvement gradually slows down, and its performance eventually cannot exceed a theoretical limit called the **Bayes Optimal Error**.

The Bayes Optimal Error is considered the theoretically achievable optimal error. In other words, it represents the theoretically optimal function, and no function mapping from x to accuracy y can surpass this value. For example, in speech recognition, some audio segments are so noisy that it's practically impossible to determine what is being said, so perfect recognition accuracy cannot reach 100%.

Since human performance is very close to the Bayes Optimal Error for some natural perception tasks, once a machine learning system's performance surpasses human performance, there is not much room for further improvement.

## Summary

To make a supervised learning algorithm reach a level of usability, two key aspects should be achieved:

1. The algorithm fits well to the training set, indicating low bias.
2. It generalizes effectively to the validation and test sets, meaning low variance.

![Human-level](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Structuring_Machine_Learning_Projects/Human-level.png)

# Introduction to ML Strategy II
(skip for now)

# Convolutional Neural Networks

Some very good resources that introduce CNN: [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
Also this [Chinese Article](https://www.zhihu.com/question/34681168/answer/2359313510)

## Padding in Convolution

Assuming the input image size is $n \times n$, and the filter size is $f \times f$, the size of the output image after convolution is $(n-f+1) \times (n-f+1)$.

This leads to two issues:

* After each convolution operation, the size of the output image reduces.
* The pixels in the corners and edge regions of the original image contribute less to the output, causing the output image to lose a lot of edge information.

To address these problems, padding can be applied to the original image at the boundaries before performing the convolution operation, effectively increasing the matrix size. Typically, 0 is used as the padding value.

![Padding](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Padding.jpg)

Let the number of pixels to extend in each direction be $p$. Then, after padding, the size of the original image becomes $(n+2p) \times (n+2p)$, while the filter size remains $f \times f$. Consequently, the size of the output image becomes $(n+2p-f+1) \times (n+2p-f+1)$.

Therefore, during convolutional operations, we have two options:

* **Valid Convolution**: No padding, direct convolution. The result size is $(n-f+1) \times (n-f+1)$.
* **Same Convolution**: Padding is applied to make the output size the same as the input, ensuring $p = \frac{f-1}{2}$.

In the field of computer vision, $f$ is usually chosen as an odd number. The reasons include that in Same Convolution, $p = \frac{f-1}{2}$ yields a natural number result, and using an odd-sized filter has a center point that conveniently represents its position.

## Strided Convolution 

During the convolution process, sometimes we need to avoid information loss through padding, and sometimes we need to compress some information by setting the **Stride**.

Stride represents the distance the filter moves in the horizontal and vertical directions of the original image during convolution. Previously, the stride was often set to 1 by default. If we set the stride to 2, the convolution process looks like the image below:

![Stride](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Stride.jpg)

Let the stride be $s$, the padding length be $p$, the input image size be $n \times n$, and the filter size be $f \times f$. The size of the output image after convolution is calculated as:

$$\biggl\lfloor \frac{n+2p-f}{s}+1   \biggr\rfloor \times \biggl\lfloor \frac{n+2p-f}{s}+1 \biggr\rfloor$$

Note that there is a floor function in the formula, used to handle cases where the quotient is not an integer. Flooring reflects the condition that the blue frame, when taken over the original matrix, must be completely included within the image for computation.

So far, what we have learned as "convolution" is actually called **cross-correlation** and is not the mathematical convolution. In a true convolution operation, before performing element-wise multiplication and summation, the filter needs to be flipped along the horizontal and vertical axes (equivalent to a 180-degree rotation). However, this flip has little effect on filters that are generally horizontally or vertically symmetrical. Following the convention in machine learning, we typically do not perform this flip to simplify the code while ensuring the neural network works correctly.

## Convolution in High Dimensions

When performing convolution on RGB images with three channels, the corresponding set of filters is also three channels. The process involves convolving each individual channel (R, G, B) with its respective filter, summing the results, and then adding the sums of the three channels to get a single pixel value for the output image. This results in a total of 27 multiplications and summations to compute one pixel value.

![Convolutions-on-RGB-image](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Convolutions-on-RGB-image.png)

Different channels can have different filters. For example, if you want to detect vertical edges in the R channel only, and no edge detection in the G and B channels, you would set all the filters for the G and B channels to zero. When dealing with inputs of specific height, width, and channel dimensions, filters can have different heights and widths, but the number of channels must match the input.

If you want to simultaneously detect vertical and horizontal edges, or perform more complex edge detection, you can add more filter sets. For instance, you can set the first filter set to perform vertical edge detection and the second filter set to perform horizontal edge detection. If the input image has dimensions of $n \times n \times n\_c$ (where $n\_c$ is the number of channels) and the filter dimensions are $f \times f \times n\_c$, the resulting output image will have dimensions of $(n-f+1) \times (n-f+1) \times n'\_c$, where $n'\_c$ is the number of filter sets.

![More-Filters](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/More-Filters.jpg)

## Single-Layer Convolutional Network

![One-Layer-of-a-Convolutional-Network](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/One-Layer-of-a-Convolutional-Network.jpg)

Compared to previous convolution processes, a single layer in a convolutional neural network (CNN) introduces activation functions and biases. In contrast to the standard neural network equations:

$$Z^{[l]} = W^{[l]}A^{[l-1]}+b$$
$$A^{[l]} = g^{[l]}(Z^{[l]})$$

In CNNs, the role of the filter is analogous to the weights $W^{[l]}$, and the convolution operation replaces the matrix multiplication between $W^{[l]}$ and $A^{[l-1]}$. The activation function is typically ReLU.

For a 3x3x3 filter, including the bias $b$, there are a total of 28 parameters. Regardless of the input image's size, when using this single filter for feature extraction, the number of parameters remains fixed at 28. This means that the number of parameters in a CNN is much smaller compared to a standard neural network, which is one of the advantages of CNNs.

### Summary of Notation

Let layer $l$ be a convolutional layer:

* $f^{[l]}$: **Filter height (or width)**
* $p^{[l]}$: **Padding length**
* $s^{[l]}$: **Stride**
* $n^{[l]}_c$: **Number of filter groups**

* **Input dimensions**: $n^{[l-1]}_H \times n^{[l-1]}_W \times n^{[l-1]}_c$. Here, $n^{[l-1]}_H$ represents the height of the input image, and $n^{[l-1]}_W$ represents its width. In previous examples, the input images had the same height and width, but in practice, they might differ, so subscripts are used to differentiate them.

* **Output dimensions**: $n^{[l]}_H \times n^{[l]}_W \times n^{[l]}_c$. Where:

$$n^{[l]}_H = \biggl\lfloor \frac{n^{[l-1]}_H+2p^{[l]}-f^{[l]}}{s^{[l]}}+1   \biggr\rfloor$$

$$n^{[l]}_W = \biggl\lfloor \frac{n^{[l-1]}_W+2p^{[l]}-f^{[l]}}{s^{[l]}}+1   \biggr\rfloor$$

* **Dimensions of each filter group**: $f^{[l]} \times f^{[l]} \times n^{[l-1]}_c$. Here, $n^{[l-1]}_c$ represents the number of channels (depth) in the input image.
* **Dimensions of weights**: $f^{[l]} \times f^{[l]} \times n^{[l-1]}_c \times n^{[l]}_c$
* **Dimensions of biases**: $1 \times 1 \times 1 \times n^{[l]}_c$
