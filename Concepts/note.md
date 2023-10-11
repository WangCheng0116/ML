# Table of Contents
<!-- toc -->
- [Table of Contents](#table-of-contents)
- [Linear Regression with one variable](#linear-regression-with-one-variable)
  - [Gradient Descent](#gradient-descent)
  - [Apply Gradient Descent into linear regression model](#apply-gradient-descent-into-linear-regression-model)
- [Matrix Review](#matrix-review)
- [Linear Regression with multiple variables](#linear-regression-with-multiple-variables)
  - [New notations](#new-notations)
  - [Gradient Descent for Multiple Variables](#gradient-descent-for-multiple-variables)
  - [Python Implementation](#python-implementation)
  - [Feature Scaling](#feature-scaling)
  - [Learning Rate](#learning-rate)
- [Features and Polynomial Regression](#features-and-polynomial-regression)
- [Normal Equation](#normal-equation)
- [Comparison](#comparison)
  - [What if X^T X is not invertible?](#what-if-xt-x-is-not-invertible)
- [Logistic Regression](#logistic-regression)
  - [Classification Problems](#classification-problems)
  - [Decision Boundary](#decision-boundary)
  - [Cost Function](#cost-function)
  - [Vectorization of Logistic Regression](#vectorization-of-logistic-regression)
  - [Multiclass Classification](#multiclass-classification)
- [Regularization](#regularization)
  - [Underfit and Overfit](#underfit-and-overfit)
  - [Motivation to have regularization](#motivation-to-have-regularization)
  - [Gradient Descent and Normal Equation Using Regularization in Linear Regression](#gradient-descent-and-normal-equation-using-regularization-in-linear-regression)
  - [Gradient Descent Using Regularization in Logistic Regression](#gradient-descent-using-regularization-in-logistic-regression)
  - [Gradient Descent Using Regularization in Neural Networks](#gradient-descent-using-regularization-in-neural-networks)
  - [Dropout](#dropout)
    - [Implementation](#implementation)
  - [Other Methods of Regularization](#other-methods-of-regularization)
- [Neural Networks](#neural-networks)
  - [Model Representation I](#model-representation-i)
  - [Other activation functions](#other-activation-functions)
  - [Multiclass Classification](#multiclass-classification-1)
  - [Cost Function](#cost-function-1)
  - [Backpropagation Algorithm](#backpropagation-algorithm)
  - [Randomize Initialization](#randomize-initialization)
- [Deep Neural Network](#deep-neural-network)
  - [Working Flow Chart](#working-flow-chart)
- [Setting Up ML Application](#setting-up-ml-application)
  - [Data Split: Training / Validation / Test Sets](#data-split-training--validation--test-sets)
  - [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
  - [Gradient Checking (Grad Check)](#gradient-checking-grad-check)
- [Optimization Algorithms](#optimization-algorithms)
  - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
    - [Batch Gradient Descent:](#batch-gradient-descent)
    - [Stochastic Gradient Descent:](#stochastic-gradient-descent)
  - [Exponentially Weighted Averages](#exponentially-weighted-averages)
  - [Bias Correction in Exponentially Weighted Averages](#bias-correction-in-exponentially-weighted-averages)
  - [Gradient Descent with Momentum](#gradient-descent-with-momentum)
  - [RMSprop(Root Mean Square Propagation)](#rmsproproot-mean-square-propagation)
  - [Adam Optimization Algorithm(Adaptive Moment Estimation)](#adam-optimization-algorithmadaptive-moment-estimation)
  - [Learning Rate Decay](#learning-rate-decay)
  - [Local Optima](#local-optima)
- [Hyperparameter Tuning Batch Normalization and Programming Frameworks](#hyperparameter-tuning-batch-normalization-and-programming-frameworks)
  - [Tuning Process](#tuning-process)
  - [Using an Appropriate Scale to Pick Hyperparameters](#using-an-appropriate-scale-to-pick-hyperparameters)
  - [Hyperparameters Tuning in Practice: Pandas vs. Caviar](#hyperparameters-tuning-in-practice-pandas-vs-caviar)
  - [Normalizing Activations in a Network](#normalizing-activations-in-a-network)
  - [Batch Normalization (BN)](#batch-normalization-bn)
  - [Fitting Batch Norm into a Neural Network](#fitting-batch-norm-into-a-neural-network)
  - [Batch Normalization and Its Usage](#batch-normalization-and-its-usage)
  - [Why Does Batch Norm Work?](#why-does-batch-norm-work)
  - [Benefits of Batch Normalization](#benefits-of-batch-normalization)
  - [Batch Norm at Test Time](#batch-norm-at-test-time)
  - [Softmax Regression](#softmax-regression)
- [Introduction to ML Strategy I](#introduction-to-ml-strategy-i)
  - [Orthogonalization in Machine Learning](#orthogonalization-in-machine-learning)
  - [Training / Validation / Test Set Split](#training--validation--test-set-split)
  - [Distribution of Validation and Test Sets](#distribution-of-validation-and-test-sets)
  - [Partitioning Data](#partitioning-data)
  - [Comparing to Human Performance](#comparing-to-human-performance)
  - [Summary](#summary)
- [Introduction to ML Strategy II](#introduction-to-ml-strategy-ii)
- [Convolutional Neural Networks](#convolutional-neural-networks)
  - [Padding in Convolution](#padding-in-convolution)
  - [Strided Convolution](#strided-convolution)
  - [Convolution in High Dimensions](#convolution-in-high-dimensions)
  - [Single-Layer Convolutional Network](#single-layer-convolutional-network)
    - [Summary of Notation](#summary-of-notation)
  - [Simple Convolutional Network Example](#simple-convolutional-network-example)
  - [Pooling Layer](#pooling-layer)
  - [Fully Connected Layer](#fully-connected-layer)
  - [Example of a Convolutional Neural Network (CNN)](#example-of-a-convolutional-neural-network-cnn)
  - [Why Convolutions?](#why-convolutions)
- [Convolutional Neural Networks (CNN) Architectures](#convolutional-neural-networks-cnn-architectures)
  - [LeNet-5](#lenet-5)
  - [AlexNet](#alexnet)
  - [VGG](#vgg)
  - [ResNet (Residual Network)](#resnet-residual-network)
    - [Reasons Why Residual Networks Work](#reasons-why-residual-networks-work)
- [Object Detection](#object-detection)
  - [Object Localization](#object-localization)
  - [Object Detection](#object-detection-1)
  - [Convolutional Implementation of Sliding Windows (Do partition all at once, no need to feed in sliding windows sequentially)](#convolutional-implementation-of-sliding-windows-do-partition-all-at-once-no-need-to-feed-in-sliding-windows-sequentially)
  - [Bounding Box Predictions](#bounding-box-predictions)
  - [Intersection Over Union (IoU)](#intersection-over-union-iou)
  - [Non-Maximum Suppression](#non-maximum-suppression)
  - [R-CNN](#r-cnn)
- [Face Recognition and Neural Style Transfer](#face-recognition-and-neural-style-transfer)
  - [Difference Between Face Verification and Face Recognition](#difference-between-face-verification-and-face-recognition)
  - [One-Shot Learning](#one-shot-learning)
  - [Siamese Network](#siamese-network)
  - [Triplet Loss](#triplet-loss)
  - [As Binary Classification](#as-binary-classification)
  - [Neural Style Transfer](#neural-style-transfer)
  - [What Deep Convolutional Networks Learn](#what-deep-convolutional-networks-learn)
  - [Cost Function For Neural Style Transfer](#cost-function-for-neural-style-transfer)
    - [Content Cost Function](#content-cost-function)
    - [Style Cost Function](#style-cost-function)
- [Sequence Models](#sequence-models)
  - [Notations](#notations)
  - [Recurrent Neural Network (RNN) Model](#recurrent-neural-network-rnn-model)
    - [Forward Propagation](#forward-propagation)
    - [Backpropagation Through Time (BPTT)](#backpropagation-through-time-bptt)
  - [Different Types of RNNs](#different-types-of-rnns)
  - [Language Model](#language-model)
  - [Sampling](#sampling)
  - [The Gradient Vanishing Problem in RNNs](#the-gradient-vanishing-problem-in-rnns)
- [K Nearest Neighbors (KNN) Algorithm](#k-nearest-neighbors-knn-algorithm)
- [Random Forest](#random-forest)



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

Some very good resources that introduce CNN:  
[A Zhihu Article](https://www.zhihu.com/question/34681168/answer/2359313510), with detailed illustration. Can read this instead of the following notes.   
[A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)


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

## Simple Convolutional Network Example

A simple CNN model is depicted in the following diagram:

![Simple-Convolutional-Network-Example](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Simple-Convolutional-Network-Example.jpg)

In this model, $a^{[3]}$ has dimensions 7x7x40. It flattens the 1960 features into a column vector of 1960 units, which is then connected to the output layer. The output layer can consist of a single neuron for binary classification (logistic) or multiple neurons for multi-class classification (softmax). Finally, it produces the predicted output $\hat y$.

As the depth of the neural network increases, the height and width of the images ($n^{[l]}_H$ and $n^{[l]}_W$) generally decrease, while $n^{[l]}_c$ increases.

A typical convolutional neural network usually consists of three types of layers: **Convolutional Layer**, **Pooling Layer**, and **Fully Connected Layer(FC)**. While it's possible to build a good neural network with just convolutional layers, most networks also incorporate pooling and fully connected layers, as they are easier to design.

## Pooling Layer

The **pooling layer** serves to reduce the size of the model, increase computational speed, and enhance the robustness of extracted features by downsampling the input.

* One common pooling technique is called **Max Pooling**, where the input is divided into different regions, and each element in the output is the maximum value within the corresponding region, as shown below:
![Max-Pooling](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Max-Pooling.png)
The pooling process is similar to the convolution process. In the example above, a filter with a size of $f=2$ and a stride of $s=2$ is applied, which means it moves in steps of 2. 

* Another pooling technique is **Average Pooling**, which computes the average value of elements within a region instead of taking the maximum.  
  
Pooling layers have a set of hyperparameters, including filter size $f$, stride $s$, and the choice of max or average pooling. Unlike convolution layers, pooling layers do not have parameters that need to be learned or modified during training. Once the hyperparameters are set, we do not need to adjust them.

The input dimensions for pooling are:

$$n_H \times n_W \times n_c$$

The output dimensions are calculated as:

$$\biggl\lfloor \frac{n_H-f}{s}+1 \biggr\rfloor \times \biggl\lfloor \frac{n_W-f}{s}+1 \biggr\rfloor \times n_c$$
Notice: pooling layer does not change the number of channels.

## Fully Connected Layer
A fully connected layer in a Convolutional Neural Network (CNN), also known as a dense layer or fully connected neural network, is a traditional neural network layer in which every neuron is connected to every neuron in the previous and subsequent layers. This type of layer is commonly used in the latter part of a CNN for high-level reasoning. It also has the same parameters as a standard neural network layer, like weights and biases.

## Example of a Convolutional Neural Network (CNN)

![CNN-Example](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/CNN-Example.jpg)

When counting the number of layers in a neural network, it's common to consider layers with weights and parameters. Therefore, pooling layers are typically counted as one layer along with the preceding convolutional layers.

The diagram shows an example of a CNN architecture, including convolutional layers (CONV), pooling layers (POOL), and fully connected layers (FC). The dimensions and parameters of each layer are summarized in the table below:

![Alt text](image-7.png)

## Why Convolutions?
* Parameter sharing: A feature detector (such as a vertical edge detector) that is useful in one part of the image is probably useful in another part of the image.
* Sparsity of connections: In each layer, each output value depends only on a small number of inputs.  
  
Can refer to this section's title area to find recommended resources to further understand the above two points.

# Convolutional Neural Networks (CNN) Architectures

## LeNet-5

![LeNet-5](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/LeNet-5.png)

**Features:**

- LeNet-5 is designed for grayscale images, so it takes input images with 1 channel.
- The model contains approximately 60,000 parameters, which is much fewer compared to standard neural networks.
- The typical LeNet-5 architecture includes Convolutional layers (CONV), Pooling layers (POOL), and Fully Connected layers (FC), arranged in the order of CONV->POOL->CONV->POOL->FC->FC->OUTPUT. The pattern of one or more convolutional layers followed by a pooling layer is still widely used.
- When LeNet-5 was proposed, it used average pooling and often employed Sigmoid and tanh as activation functions. Nowadays, improvements such as using max-pooling and ReLU as activation functions are common.

**Related Paper:** [LeCun et al., 1998. Gradient-based learning applied to document recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791&tag=1). Professor Andrew Ng recommends a detailed reading of the second paragraph and a general reading of the third paragraph.

---

## AlexNet

![AlexNet](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/AlexNet.png)

**Features:**

- AlexNet is similar to LeNet-5 but more complex, containing about 60 million parameters. Additionally, AlexNet uses the ReLU function.
- When used to train image and data sets, AlexNet can handle very similar basic building blocks, often comprising a large number of hidden units or data.

**Related Paper:** [Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf). This paper is easy to understand and has had a significant impact, marking the beginning of the deep learning era in computer vision.

---

## VGG

![VGG](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/VGG.png)

**Features:**

- VGG, also known as VGG-16, refers to a network with 16 convolutional and fully connected layers.
- It has fewer hyperparameters to focus on, mainly concerning the construction of convolutional layers.
- The structure is not overly complex and is regular, doubling the number of filters in each group of convolutional layers.
- VGG requires a vast number of trainable features, with as many as approximately 138 million parameters.

**Related Paper:** [Simonyan & Zisserman 2015. Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf).

## ResNet (Residual Network)

![Residual-block](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Residual-block.jpg)

The existence of gradient vanishing and exploding problems makes it increasingly difficult to train networks as they become deeper. **Residual Networks (ResNets)** effectively address this issue.

The structure shown above is called a **residual block**. Through **shortcut connections**, $a^{[l]}$ can be added to the second ReLU process, establishing a direct connection between $a^{[l]}$ and $a^{[l+2]}$. The expressions are as follows:

$$z^{[l+1]} = W^{[l+1]}a^{[l]} + b^{[l+1]}$$

$$a^{[l+1]} = g(z^{[l+1]})$$

$$z^{[l+2]} = W^{[l+2]}a^{[l+1]} + b^{[l+2]}$$

$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$

Building a residual network involves stacking many residual blocks together to create a deep network.

![Residual-Network](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Residual-Network.jpg)

To distinguish, in the ResNets paper by [He et al., 2015. Deep residual networks for image recognition](https://arxiv.org/pdf/1512.03385.pdf), non-residual networks are referred to as **plain networks**. The method to transform them into residual networks is to add all the skip connections.

In theory, as the network depth increases, performance should improve. However, in practice, for a plain network, as the network depth increases, training error decreases initially, then starts to increase. But the training results of Residual Networks show that even as the network becomes deeper, its performance on the training set continues to improve.

![ResNet-Training-Error](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/ResNet-Training-Error.jpg)

Residual networks help address the issues of gradient vanishing and exploding, allowing for the training of much deeper networks while maintaining good performance.

### Reasons Why Residual Networks Work

Let's consider a large neural network with an input $X$ and an output $a^{[l]}$. We'll add two extra layers to this network, resulting in an output $a^{[l+2]}$. We'll treat these two layers as a residual block with a skip connection. For the sake of explanation, let's assume that ReLU is used as the activation function throughout the network, ensuring that all activation values are greater than or equal to 0.

![Why-do-residual-networks-work](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Why-do-residual-networks-work.jpg)

Then, we have:

$$
\begin{equation}
\begin{split}
 a^{[l+2]} &= g(z^{[l+2]}+a^{[l]})  
     \\\ &= g(W^{[l+2]}a^{[l+1]}+b^{[l+2]}+a^{[l]})
\end{split}
\end{equation}
$$

When gradient vanishing occurs, $W^{[l+2]}\approx0$ and $b^{[l+2]}\approx0$. Therefore:

$$a^{[l+2]} = g(a^{[l]}) = ReLU(a^{[l]}) = a^{[l]}$$

Thus, these two additional residual block layers do not degrade the network's performance. However, when gradient vanishing does not occur, the learned non-linear relationships further enhance the performance.


# Object Detection
## Object Localization

In classification tasks, it is not only necessary to identify the category of objects in an image but also to precisely locate them by drawing **Bounding Boxes** around them. Typically, in classification tasks, there is usually a single large object located in the center of the image. In contrast, in object detection tasks, an image can contain multiple objects, possibly of different classes, in a single frame.

To determine the location of a car in an image, a neural network can output four numbers denoted as $b_x$, $b_y$, $b_h$, and $b_w$. If we consider the top-left corner of the image as (0, 0) and the bottom-right corner as (1, 1), then the following relationships apply:

* Center of the red bounding box: ($b_x$, $b_y$)
* Height of the bounding box: $b_h$
* Width of the bounding box: $b_w$

Therefore, the training dataset should not only include object classification labels but also four numbers representing the bounding box. The target label $Y$ can be defined as follows:

$$
Y = \begin{bmatrix}
P_c \\
b_x \\
b_y \\
b_h \\
b_w \\
c_1 \\
c_2 \\
c_3
\end{bmatrix}
$$

Where:

- $P_c=1$ indicates the presence of an object.
- $c_n$ represents the probability of the object belonging to class $n$.
- If $P_c=0$, it means no object is detected, and the values of the remaining 7 parameters are irrelevant and can be ignored (denoted as "?").

$$
P_c=0, Y = \begin{bmatrix}
0 \\
? \\
? \\
? \\
? \\
? \\
? \\
?
\end{bmatrix}
$$

The loss function can be defined as $L(\hat y, y)$. If the squared error form is used, different loss functions apply to different values of $P\_c$ (Note: the subscript $i$ denotes the $i$-th value in the label):

1. When $P\_c=1$, i.e., $y\_1=1$:

    $L(\hat y, y) = (\hat y_1 - y_1)^2 + (\hat y_2 - y_2)^2 + \cdots + (\hat y_8 - y_8)^2$

2. When $P\_c=0$, i.e., $y\_1=0$:

    $L(\hat y, y) = (\hat y_1 - y_1)^2$

Besides using squared error, logistic regression loss can also be employed, and class labels $c_1$, $c_2$, $c_3$ can be output using softmax. However, squared error is usually sufficient to achieve good results.

## Object Detection

To achieve object detection, we can use the **Sliding Windows Detection** algorithm. The steps of this algorithm are as follows:

1. Gather various target and non-target images from the training dataset. Sample images should be relatively small in size, with the corresponding target object located at the center and occupying a substantial portion of the entire image.

2. Build a CNN model using the training dataset to ensure the model has a high recognition rate.

3. Choose an appropriately sized window and an appropriate fixed stride for sliding from left to right and top to bottom of the test image. Use the pre-trained CNN model to make recognition judgments for each window region.

4. Optionally, you can select larger windows and repeat the operation from the third step.

![Sliding-windows-detection](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Sliding-windows-detection.png)

The **advantage** of sliding windows detection is its simplicity in principle and the fact that it doesn't require manually selecting target areas. However, the **disadvantages** include the need to manually set the window size and stride intuitively. Choosing windows or strides that are too small or too large can reduce the accuracy of object detection. Additionally, performing CNN computations for each sliding operation can be computationally expensive, especially with small windows and strides.

Therefore, while sliding windows object detection is straightforward, its performance is suboptimal, and it can be less efficient.

## Convolutional Implementation of Sliding Windows (Do partition all at once, no need to feed in sliding windows sequentially)
![Convolution-implementation-of-sliding-windows](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Convolution-implementation-of-sliding-windows.png)

As shown in the image, for a 16x16x3 image with a stride of 2, the CNN network produces an output layer of 2x2x4. Here, 2x2 represents a total of 4 window results. For a more complex 28x28x3 image, the output layer becomes 8x8x4, resulting in a total of 64 window results. The max-pooling layer has equal width and height as well as equal strides.

The principle behind the speed improvement is that during the sliding window process, CNN forward computations need to be repeated. Instead of dividing the input image into multiple subsets and performing forward propagation separately, the entire image is input to the convolutional network for a single CNN forward computation. This allows for the sharing of calculations in common regions, reducing computational costs.

Related Paper: [Sermanet et al., 2014. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks](https://arxiv.org/pdf/1312.6229.pdf)

## Bounding Box Predictions

In the previous algorithm, the position of bounding boxes may not perfectly cover the target, or their size may not be suitable, or the most accurate bounding box might not be a square but a rectangle.

The **YOLO (You Only Look Once) algorithm** can be used to obtain more accurate bounding boxes. The YOLO algorithm divides the original image into an n×n grid and applies the image classification and object localization algorithms mentioned in the Object Localization (aforementioned chapter) section to each grid individually. Each grid has labels like:

$$\left[\begin{matrix}P_c\\\ b_x\\\ b_y\\\ b_h\\\ b_w\\\ c_1\\\ c_2\\\ c_3\end{matrix}\right]$$

If the center of a certain target falls within a grid, that grid is responsible for detecting that object.

![Bounding-Box-Predictions](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Bounding-Box-Predictions.png)

As shown in the example above, if the input image is divided into a 3×3 grid and there are 3 classes of objects to detect, the label for each part of the image within the grid will be an 8-dimensional column matrix. The final output will be of size 3×3×8. To obtain this result, a CNN with an input size of 100×100×3 and an output size of 3×3×8 needs to be trained. In practice, a finer 19×19 grid may be used, making it less likely for the centers of two targets to fall in the same grid.

Advantages of the YOLO algorithm:

1. Similar to image classification and object localization algorithms, it explicitly outputs bounding box coordinates and sizes, not limited by the stride size of the sliding window classifier.
2. It still performs only one CNN forward computation, making it highly efficient and even capable of real-time recognition.

How are bounding boxes $b_x$, $b_y$, $b_h$, and $b_w$ encoded? The YOLO algorithm sets the values of $b_x$, $b_y$, $b_h$, and $b_w$ as proportions relative to the grid length. Therefore, $b_x$ and $b_y$ are between 0 and 1, while $b_h$ and $b_w$ can be greater than 1. Of course, there are other parameterization forms, and they may work even better. This is just a general representation.

Related Paper: [Redmon et al., 2015. You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf). Ng considers this paper somewhat difficult to understand.

## Intersection Over Union (IoU)

The **Intersection Over Union (IoU)** function is used to evaluate object detection algorithms. It calculates the ratio of the intersection (I) of the predicted bounding box and the actual bounding box to their union (U):

$$IoU = \frac{I}{U}$$

The IoU value ranges from 0 to 1, with values closer to 1 indicating more accurate object localization. When IoU is greater than or equal to 0.5, it is generally considered that the predicted bounding box is correct, although a higher threshold can be chosen for stricter criteria.


## Non-Maximum Suppression

In the YOLO algorithm, it's possible that multiple grids detect the same object. **Non-Maximum Suppression (NMS)** is used to clean up detection results and find the grid where the center of each object is located, ensuring that the algorithm detects each object only once.

The steps for performing non-maximum suppression are as follows:

1. Discard grids with confidence $P_c$ (which contains the center of the object) less than a threshold (e.g., 0.6).
2. Select the grid with the highest $P_c$.
3. Calculate the Intersection over Union (IoU) between this grid and all other grids and discard grids with IoU exceeding a predetermined threshold.
4. Repeat steps 2-3 until there are no unprocessed grids left.

The above steps are suitable for single-class object detection. For multi-class object detection, non-maximum suppression should be performed separately for each class.

## R-CNN

The sliding window object detection algorithm introduced earlier scans regions even in areas where there are clearly no objects, which reduces the efficiency of the algorithm. To address this issue, **R-CNN (Region CNN, Region-based Convolutional Neural Network)** was proposed. It uses an **image segmentation algorithm** to identify **candidate regions (Region Proposals)** on different colored blocks within the input image. The classification is then performed only on these regions.

![R-CNN](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/R-CNN.png)

The drawback of R-CNN is that it is slow in terms of processing speed. Therefore, a series of subsequent research efforts aimed to improve it. For example, Fast R-CNN (which is similar to convolution-based sliding window implementations but still has a slow region proposal step) and Faster R-CNN (which uses convolution to segment the image). However, in most cases, they are still slower than the YOLO algorithm.

Related research papers:

- R-CNN: [Girshik et al., 2013. Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/pdf/1311.2524.pdf)
- Fast R-CNN: [Girshik, 2015. Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- Faster R-CNN: [Ren et al., 2016. Faster R-CNN: Towards real-time object detection with region proposal networks](https://arxiv.org/pdf/1506.01497v3.pdf)

# Face Recognition and Neural Style Transfer

## Difference Between Face Verification and Face Recognition

* Face Verification:
Face verification typically refers to a one-to-one problem. It involves verifying whether an input facial image matches a known identity.

* Face Recognition:
Face recognition is a more complex one-to-many problem. It involves determining whether an input facial image matches any of the known identity information from a group of multiple identities.

In general, face recognition is more challenging compared to face verification due to the increased number of potential identity matches, which can lead to higher error rates.

## One-Shot Learning
Suppose a company has 4 emploees, so in this case we don't have enough sample to train our model to gain sufficient robustness. Instead, we need to opt for anohter approach, which is **One-Shot**.

Therefore, we implement the One-Shot Learning process by learning a **Similarity** function. The Similarity function quantifies the dissimilarity between two input images, and its formula is defined as follows:

$$Similarity = d(img1, img2)$$

## Siamese Network

One way to implement the Similarity function is by using a **Siamese network**, which is a type of neural network that runs the same convolutional network on two different inputs and then compares their outputs.

![Siamese](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Siamese.png)

As shown in the above example, two images, $x^{(1)}$ and $x^{(2)}$, are separately fed into two identical convolutional networks. After passing through fully connected layers, instead of applying a Softmax layer, they produce feature vectors $f(x^{(1)})$ and $f(x^{(2)})$. The Similarity function is then defined as the L2 norm of the difference between these two feature vectors:

$$d(x^{(1)}, x^{(2)}) = ||f(x^{(1)}) - f(x^{(2)})||^2_2$$

Reference Paper: [Taigman et al., 2014, DeepFace: Closing the Gap to Human-Level Performance](http://www.cs.wayne.edu/~mdong/taigman_cvpr14.pdf)

## Triplet Loss

The **Triplet loss function** is used to train parameters that result in high-quality encodings of facial images. The term "Triplet" comes from the fact that training this neural network requires a large dataset containing sets of images with Anchor, Positive, and Negative examples.
> You can think it this way: Anchor is the image of a person, Positive is the image of the same person, and Negative is the image of a different person.

![Training-set-using-triplet-loss](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Training-set-using-triplet-loss.png)

For these three images, the following condition should hold:

$$||f(A) - f(P)||^2_2 + \alpha \leq ||f(A) - f(N)||^2_2$$

Here, $\alpha$ is referred to as the **margin**, and it ensures that $f()$ does not always output zero vectors (or a constant value).

The definition of the Triplet loss function is as follows:

$$L(A, P, N) = \max(||f(A) - f(P)||^2_2 - ||f(A) - f(N)||^2_2 + \alpha, 0)$$

The max function is used because the value of $||f(A) - f(P)||^2_2 - ||f(A) - f(N)||^2_2 + \alpha$ should be less than or equal to 0, and taking the maximum with 0 ensures this constraint.

For a training set of size $m$, the cost function is defined as:

$$J = \sum^{m}_{i=1}L(A^{(i)}, P^{(i)}, N^{(i)})$$

The goal is to minimize this cost function using gradient descent.

When selecting training samples, random selection can lead to Anchor and Positive examples being very similar while Anchor and Negative examples are very dissimilar, making it challenging for the model to capture key differences. **Therefore, it's better to artificially increase the difference between Anchor and Positive examples while reducing the difference between Anchor and Negative examples, encouraging the model to learn critical distinctions between different faces**.
> Feed in hard to distinguish examples to the model.

Reference Paper: [Schroff et al., 2015, FaceNet: A unified embedding for face recognition and clustering](https://arxiv.org/pdf/1503.03832.pdf)

## As Binary Classification 

In addition to the Triplet loss function, a binary classification structure can also be used to learn parameters for solving the face recognition problem. The approach involves inputting a pair of images and passing the feature vectors generated by two Siamese networks to the same Sigmoid unit. An output of 1 indicates that the two images are recognized as the same person, while an output of 0 indicates that they are recognized as different individuals.

The expression corresponding to the Sigmoid unit is as follows:

$$\hat y = \sigma \left(\sum^K_{k=1}w_k|f(x^{(i)})_k - f(x^{(j)})_k| + b\right)$$

Here, $w_k$ and $b$ are parameters obtained through iterative training using gradient descent. The calculation expression above can also be replaced by an alternative expression:

$$\hat y = \sigma \left(\sum^K_{k=1}w_k \frac{(f(x^{(i)})_k - f(x^{(j)})_k)^2}{f(x^{(i)})_k + f(x^{(j)})_k} + b\right)$$

In this alternative expression, $\frac{(f(x^{(i)})_k - f(x^{(j)})_k)^2}{f(x^{(i)})_k + f(x^{(j)})_k}$ is referred to as the $\chi$-square similarity.
 
Reference Paper: [Sun et al., 2014, Deep Learning Face Representation by Joint Identification-Verification](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Sun_Deep_Learning_Face_2014_CVPR_paper.pdf)

## Neural Style Transfer

**Neural style transfer** involves "transferring" the style of a reference image to another content image to generate an image with its unique characteristics.

![Neural-style-transfer](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Neural-style-transfer.png)

## What Deep Convolutional Networks Learn

To understand how neural style transfer works, it's essential to comprehend what a deep convolutional network learns from input image data. Visualization can help us achieve this understanding.

![Visualizing-deep-layers](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/Visualizing-deep-layers.png)

By iterating through all training samples, we can identify the 9 image regions that maximize the output of the activation function for each layer. It's evident that shallow layers typically detect simple features such as edges, colors, and shadows in the original image. **As the depth of the network increases, hidden units can capture larger regions and learn features that range from edges to textures and then to specific objects, becoming more complex.**

Reference Paper: [Zeiler and Fergus., 2013, Visualizing and understanding convolutional networks](https://arxiv.org/pdf/1311.2901.pdf)

## Cost Function For Neural Style Transfer

The cost function for generating an image G in neural style transfer is defined as follows:

$$J(G) = \alpha \cdot J_{content}(C, G) + \beta \cdot J_{style}(S, G)$$

Here, $\alpha$ and $\beta$ are hyperparameters used to control the weighting of content and style similarity.

The algorithm steps for neural style transfer are as follows:

1. Randomly initialize the pixel values of the image G.
2. Use gradient descent to minimize the cost function iteratively, continually updating the pixel values of G.

Reference Paper: [Gatys al., 2015. A neural algorithm of artistic style](https://arxiv.org/pdf/1508.06576v2.pdf)

### Content Cost Function

The content cost function $J_{content}(C, G)$ measures the similarity between a content image C and a generated image G. The calculation process for $J_{content}(C, G)$ is as follows:

1. Utilize a pre-trained CNN (such as VGG).
2. Choose a hidden layer $l$ to compute the content cost. If $l$ is too small, the similarity will be at the pixel level; if $l$ is too large, it may capture only high-level objects. Therefore, $l$ is typically chosen as an intermediate layer.
3. Let $a^{(C)[l]}$ and $a^{(G)[l]}$ represent the activations of C and G at layer $l$. The content cost is then calculated as follows:

$$J_{content}(C, G) = \frac{1}{2} ||(a^{(C)[l]} - a^{(G)[l]})||^2$$

The smaller the difference between $a^{(C)[l]}$ and $a^{(G)[l]}$, the smaller the content cost $J_{content}(C, G)$, indicating higher similarity between the content of the two images.

### Style Cost Function
[It is well explained here](https://baozoulin.gitbook.io/neural-networks-and-deep-learning/di-si-men-ke-juan-ji-shen-jing-wang-luo-convolutional-neural-networks/convolutional-neural-networks/special-applications/410-feng-ge-dai-jia-han-shu-ff08-style-cost-function)

> So now we can conclude the Cost Function For Neural Style Transfer.


# Sequence Models
## Notations
For a sequence of data $x$, we use the symbol $x^{⟨t⟩}$ to represent the $t$-th element in the data, and we use $y^{⟨t⟩}$ to represent the $t$-th label. $T_x$ and $T_y$ denote the lengths of the input and output, respectively. For audio data, the elements may correspond to frames, while for a sentence, the elements may represent one or more words.

The $t$-th element of the $i$-th sequence data is denoted as $x^{(i)⟨t⟩}$, and the $t$-th label is $y^{(i)⟨t⟩}$. Therefore, we have $T^{(i)}_x$ and $T^{(i)}_y$.

To represent a word, we first need to create a Vocabulary or Dictionary. All the words to be represented are transformed into a column vector, which can be arranged alphabetically. Then, based on the position of the word in the vector, we use a one-hot vector to represent the label for that word. Each word is encoded as an $R^{|V| \times 1}$ vector, where $|V|$ is the number of words in the vocabulary. The index of a word in the vocabulary corresponds to a '1' in the vector, while all other elements are '0'.

For example, 'zebra' is at the last position in the vocabulary, so its word vector representation is:

$$w^{zebra} = \left [ 0, 0, 0, ..., 1\right ]^T$$

## Recurrent Neural Network (RNN) Model


When dealing with sequential data, standard neural networks face the following challenges:

1. For different examples, input and output sequences may have varying lengths, making it impossible to fix the number of neurons in the input and output layers.
2. Features learned from different positions in the input sequence cannot be shared.
3. The model has too many parameters, leading to high computational complexity.

To address these issues, we introduce the **Recurrent Neural Network (RNN)**. The structure of a typical RNN is illustrated in the following diagram:

![Recurrent-Neural-Network](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Sequence_Models/Recurrent-Neural-Network.png)


### Forward Propagation

In an RNN, at each time step, the element $x^{⟨t⟩}$ is fed into the hidden layer corresponding to that time step, and the hidden layer also receives activations $a^{⟨t-1⟩}$ from the previous time step. Typically, $a^{⟨0⟩}$ is initialized as a zero vector. Each time step produces a corresponding prediction $\hat y^{⟨t⟩}$.

RNNs process data from left to right, and the parameters are shared across time steps. The parameters for input, activation, and output are denoted as $W\_{ax}$, $W\_{aa}$, and $W\_{ya}$, respectively.

The structure of a single RNN cell is shown below:

![RNN-cell](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Sequence_Models/RNN-cell.png)

The forward propagation equations are as follows:

$$a^{⟨0⟩} = \vec{0}$$

$$a^{⟨t⟩} = g_1(W_{aa}a^{⟨t-1⟩} + W_{ax}x^{⟨t⟩} + b_a)$$

$$\hat y^{⟨t⟩} = g_2(W_{ya}a^{⟨t⟩} + b_y)$$

Here, $g_1$ is typically the tanh activation function (or sometimes ReLU), and $g_2$ can be either sigmoid or softmax, depending on the desired output type.

To simplify the equations further for efficient computation, you can concatenate $W_{aa}$ and $W_{ax}$ **horizontally** into a matrix $W_a$, and stack $a^{⟨t-1⟩}$ and $x^{⟨t⟩}$ into a single matrix. This results in:

$$W_a = [W\_{aa}, W\_{ax}]$$

$$a^{⟨t⟩} = g_1(W_a[a^{⟨t-1⟩}; x^{⟨t⟩}] + b_a)$$

$$\hat y^{⟨t⟩} = g_2(W_{ya}a^{⟨t⟩} + b_y)$$

### Backpropagation Through Time (BPTT)
![formula-of-RNN](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Sequence_Models/formula-of-RNN.png)

## Different Types of RNNs
![Examples-of-RNN-architectures](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Sequence_Models/Examples-of-RNN-architectures.png)

## Language Model

A **Language Model** is a mathematical abstraction that models language based on objective linguistic facts, allowing it to estimate the likelihood of elements appearing in a sequence. For instance, in a speech recognition system, a language model can calculate the probability of two phonetically similar sentences and use this information to make accurate decisions.

The construction of a language model relies on a large **corpus**, which is a collection of numerous sentences forming a text. The first step in building a language model is **tokenization**, where a dictionary is established. Then, each word in the corpus is represented by a corresponding one-hot vector. Additionally, an extra token "EOS" (End of Sentence) is added to denote the end of a sentence. Punctuation can be ignored or included in the dictionary and represented by one-hot vectors.

For special words in the corpus, such as names of people or places that may not be included in the dictionary, you can represent them with a UNK (Unique Token) tag instead of explicitly modeling each specific word.

The tokenized training dataset is used to train a Recurrent Neural Network (RNN), as illustrated in the following diagram:

![language-model-RNN-example](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Sequence_Models/language-model-RNN-example.png)

In the first time step, both the input $a^{⟨0⟩}$ and $x^{⟨1⟩}$ are zero vectors, and $\hat y^{⟨1⟩}$ is the probability distribution over the dictionary for the first word. In the second time step, the input $x^{⟨2⟩}$ consists of the first word $y^{⟨1⟩}$ (i.e., "cats") from the training sample's label and the previous layer's activation $a^{⟨1⟩}$. The output $y^{⟨2⟩}$ represents the conditional probabilities, calculated via softmax, of the next word in the dictionary given the word "cats." This process continues, and eventually, the model computes the probability of the entire sentence.

The loss function is defined as:

$$L(\hat y^{⟨t⟩}, y^{⟨t⟩}) = -\sum\_t y\_i^{⟨t⟩} \log \hat y^{⟨t⟩}$$

And the cost function is:

$$J = \sum\_t L^{⟨t⟩}(\hat y^{⟨t⟩}, y^{⟨t⟩})$$

## Sampling

After training a language model, you can gain insights into what the model has learned by **sampling** new sequences from it.

![Sampling](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Sequence_Models/Sampling.png)

In the first time step, the input $a^{⟨0⟩}$ and $x^{⟨1⟩}$ are both zero vectors. The model outputs the probabilities of each word in the dictionary as the first word. To generate the next word, we perform random sampling based on the softmax distribution (`np.random.choice`) using the probabilities obtained from the model. The sampled $\hat y^{⟨1⟩}$ is then used as the input $x^{⟨2⟩}$ for the next time step. This process continues until the EOS token is sampled. Ultimately, the model generates sentences, allowing you to discover the knowledge it has learned from the corpus.

The language model described here is based on building vocabulary. Alternatively, you can construct a character-based language model, which has the advantage of not worrying about unknown tokens (UNK). However, it can result in longer and more numerous sequences, and training costs can be high. Therefore, vocabulary-based language models are more commonly used.

## The Gradient Vanishing Problem in RNNs

In the sentences:

* "The cat, which already ate a bunch of food, was full."
* "The cats, which already ate a bunch of food, were full."

The singular or plural form of the verb in the latter part of the sentence depends on the singular or plural form of the noun in the earlier part. However, **basic RNNs struggle to capture these long-term dependencies**. The main reason behind this is the issue of gradient vanishing. During backpropagation, the error in the later layers can have a hard time affecting the computations in the earlier layers, making it difficult for the network to adjust the earlier calculations accordingly.

During backpropagation, as the number of layers increases, the gradients can not only exponentially decrease (vanishing gradient) but also exponentially increase (exploding gradient). Exploding gradients are usually easier to detect because the parameters may grow to the point of numerical overflow (leading to NaN values). In such cases, **gradient clipping** can be used to address the issue by scaling the gradient vector to ensure it doesn't become too large.

On the other hand, the problem of gradient vanishing is more challenging to tackle. Solutions like **GRUs and LSTMs** have been developed to mitigate the gradient vanishing problem in RNNs.

# K Nearest Neighbors (KNN) Algorithm
At its core, KNN classifies or predicts the target value of a new data point by examining the k-nearest data points from the training dataset. The "k" in KNN represents the number of nearest neighbors considered for making predictions. When classifying a data point, KNN counts the number of neighbors belonging to each class and assigns the class label that is most common among the k-nearest neighbors.

KNN is particularly useful when dealing with datasets where data points exhibit clustering or local patterns. It is straightforward to understand and implement, making it a valuable tool for both beginners and experienced practitioners in the field of machine learning.


# Random Forest
![Wiki](image-8.png)  
(extracted from wikipedia)  

A very great [resource](https://carbonati.github.io/posts/random-forests-from-scratch/) that I found online. I followed through the instruction and did [this](https://github.com/WangCheng0116/ML/blob/main/Code%20Implementation/Random%20Forest/random-forest-from-stratch-adapted.ipynb)

