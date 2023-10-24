# Table of Contents
<!-- toc -->
- [Table of Contents](#table-of-contents)
- [Perceptron](#perceptron)
  - [Model Description](#model-description)
  - [Geometric Interpretation](#geometric-interpretation)
  - [Loss Function](#loss-function)
  - [Gradient Descent](#gradient-descent)
  - [Algorithm](#algorithm)
  - [Convergence of Algorithm](#convergence-of-algorithm)
  - [Duality Form of Perceptron](#duality-form-of-perceptron)
  - [Duality Algorithm](#duality-algorithm)
  - [](#)
- [Decision Tree](#decision-tree)
  - [ID3 (Iterative Dichotomiser)](#id3-iterative-dichotomiser)
  - [C4.5](#c45)
  - [CART](#cart)
    - [Gini Index](#gini-index)
    - [Dealing with continuous values](#dealing-with-continuous-values)
  - [Pre-Pruning](#pre-pruning)
  - [Post-Pruning](#post-pruning)
    - [Reduced Error Pruning (REP)](#reduced-error-pruning-rep)
    - [Pessimistic Error Pruning (PEP)](#pessimistic-error-pruning-pep)
    - [Minimum Error Pruning (MEP)](#minimum-error-pruning-mep)
    - [Cost-Complexity Pruning (CCP)](#cost-complexity-pruning-ccp)
- [Bayes Classifier](#bayes-classifier)
  - [Bayes Theorem](#bayes-theorem)
  - [Naive Bayes Theorem](#naive-bayes-theorem)
  - [Bayes Classifier](#bayes-classifier-1)
  - [Native Bayes Classifier](#native-bayes-classifier)
- [Linear Regression with one variable](#linear-regression-with-one-variable)
  - [Gradient Descent](#gradient-descent-1)
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
  - [Dropout (In Neural Networks)](#dropout-in-neural-networks)
    - [Implementation](#implementation)
  - [Other Methods of Regularization](#other-methods-of-regularization)
- [Neural Networks](#neural-networks)
  - [Activation functions](#activation-functions)
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
- [Expectation Maximization (EM) Algorithm](#expectation-maximization-em-algorithm)
  - [Intuitive Explanation](#intuitive-explanation)
  - [Intuitive Algorithm](#intuitive-algorithm)
  - [Formal Proof](#formal-proof)
    - [Likelihood Function](#likelihood-function)
    - [Log Likelihood Function](#log-likelihood-function)
    - [Jensen Inequality](#jensen-inequality)
    - [EM](#em)
  - [EM Algorithm](#em-algorithm)
  - [Why EM Converges?](#why-em-converges)
- [Gaussian Mixture Model (GMM)](#gaussian-mixture-model-gmm)
  - [Notation](#notation)
  - [Algorithm](#algorithm-1)
  - [Intuitive Example](#intuitive-example)
  - [Implementation](#implementation-1)
  - [Math Proof](#math-proof)
- [Support Vector Machine](#support-vector-machine)
  - [Proof](#proof)
    - [Problem Formulation](#problem-formulation)
    - [Optimization Problem](#optimization-problem)
  - [Duality Conversion](#duality-conversion)
  - [Algorithm (HardMargin)](#algorithm-hardmargin)
  - [SoftMargin](#softmargin)
  - [Algorithm (SoftMargin)](#algorithm-softmargin)
  - [Another Perspective: HingeLoss](#another-perspective-hingeloss)
  - [SMO (Sequential Minimal Optimization)](#smo-sequential-minimal-optimization)
    - [How to choose $\\alpha\_1$ and $\\alpha\_2$ in SMO](#how-to-choose-alpha_1-and-alpha_2-in-smo)
    - [How to update $b$](#how-to-update-b)
  - [Kernel Function](#kernel-function)
    - [Positive Definite Kernel Function](#positive-definite-kernel-function)
  - [Common Kernel Functions](#common-kernel-functions)
    - [String Kernel Example](#string-kernel-example)
  - [Reference](#reference)
  - [Sklearn Usage](#sklearn-usage)
- [Ensemble Learning](#ensemble-learning)
  - [Bagging](#bagging)
    - [Random Forest](#random-forest)
  - [Boosting](#boosting)
    - [AdaBoost (Adaptive Boosting)](#adaboost-adaptive-boosting)
      - [Algorithm](#algorithm-2)
      - [Example](#example)
      - [Why $\\alpha$ is like that?](#why-alpha-is-like-that)
      - [Why we can update $w\_m$ is like that?](#why-we-can-update-w_m-is-like-that)
    - [Boosting Decision Tree](#boosting-decision-tree)
      - [Herustic Example](#herustic-example)
    - [Gradient Boosting Decision Tree](#gradient-boosting-decision-tree)
      - [Why Gradient?](#why-gradient)
      - [Algorithm](#algorithm-3)
    - [XGBoost](#xgboost)
      - [Loss Function Formulation](#loss-function-formulation)
      - [Partitioning (How to find the best split)](#partitioning-how-to-find-the-best-split)
        - [Greedy Algorithm](#greedy-algorithm)
        - [Approximate Algorithm](#approximate-algorithm)
- [K Nearest Neighbors (KNN) Algorithm](#k-nearest-neighbors-knn-algorithm)
  - [Algorithm I - Linear Scan](#algorithm-i---linear-scan)
  - [Algorithm II - KD Tree](#algorithm-ii---kd-tree)
    - [Build KD Tree](#build-kd-tree)
    - [Search KD Tree to find k nearest neighbors](#search-kd-tree-to-find-k-nearest-neighbors)

# Perceptron
## Model Description
* It's a linear classifier.
* Input: $x = (x_1, x_2, ..., x_n)$
* Output: $y = \{-1, 1\}$
* Model: $f(x) = sign(w \cdot x + b)$
* > $sign$, not $sigmoid$
* Loss Function: $L(w, b) = -\sum_{x_i \in M} y_i(w \cdot x_i + b)$

## Geometric Interpretation
* $w \cdot x + b = 0$ is a hyperplane
* $w \cdot x + b > 0$ is one side of the hyperplane
* $w \cdot x + b < 0$ is the other side of the hyperplane
* $w$ is the normal vector of the hyperplane, $b$ is the intercept.
<img src="image-38.png" width="300">

## Loss Function
If a point is misclassified, then its distance to the hyperplane is 
$$-{\frac{1}{\lVert w\rVert}}y_{i}(w\cdot x_{i}+b)$$
So total loss is: 
$$L(w, b) = -\sum_{x_i \in M} y_i(w \cdot x_i + b)$$
> $-y_i(w \cdot x_i + b)$ is always greater than $0$ for misclassified points. So loss is always not negative.
> When loss = 0, it means all points are classified correctly.
where $M$ is the set of misclassified points. 

## Gradient Descent
$$
\begin{aligned}
\frac{\partial L}{\partial w} &= -\sum_{x_i \in M} y_i x_i \\
\frac{\partial L}{\partial b} &= -\sum_{x_i \in M} y_i
\end{aligned}
$$
$$
w = w + \alpha \sum_{x_i \in M} y_i x_i \\
b = b + \alpha \sum_{x_i \in M} y_i
$$
where $\eta$ is the learning rate.
We will usually use stochastic gradient descent, which means we will update $w$ and $b$ after each misclassified point.

## Algorithm
* Initialize $w$ and $b$ as $0$
* Repeat until $loss = 0$:
  * Pick a misclassified point $(x_i, y_i)$, i.e. $y_i(w \cdot x_i + b) \leq 0$
  * Update $w$ and $b$:
    * $w = w + \alpha y_i x_i$
    * $b = b + \alpha y_i$

## Convergence of Algorithm

Suppose data is linearly separable. Then there exists a hyperplane that can classify all points correctly.

Let $\theta^* = {(w^*, b^*)}$ be the optimal solution.

Then since every point has been classified correctly, we have:

$$y_i(w^* \cdot x_i + b^*) > 0 \quad for ~i = 1, 2, ..., N$$

There exists $\gamma > 0$ such that:


$$y_i(w^* \cdot x_i + b^*) \geq \gamma \quad for ~i = 1, 2, ..., N \\
\Leftrightarrow (y_ix_i, y_i) \cdot \theta^* \geq \gamma \quad (a)$$
Suppose at step $k$, we update $w$ and $b$ as
$$
w_k \leftarrow w_{k-1} + \alpha y_ix_i \\
b_k \leftarrow b_{k-1} + \alpha y_i
$$
or equivalently, 
$$
\theta_k = \theta_{k-1} + \alpha y_i(x_i, 1) \quad (b)
$$
We can show $\theta_{k}\cdot \theta^* \geq k\alpha \gamma$, since 
$$
\begin{align}
\theta_{k}\cdot \theta^* &= (\theta_{k-1} + \alpha y_i(x_i, 1)) \cdot \theta ^* \quad(from~a)\\
&= \theta_{k-1}\cdot \theta^* + \alpha (y_ix_i, y_i) \cdot \theta ^* \\
&\geq \theta_{k-1}\cdot \theta^* + \alpha \gamma\quad (from~b) \\
&\geq \theta_{k-2}\cdot \theta^* + 2\alpha \gamma
\end{align}
$$
Solving recurstion, we have $\theta_{k}\cdot \theta^* \geq k\alpha \gamma \quad (A)$.  
 
We can also show $||\theta_k||^2  \leq k\alpha^2R^2$, where $R = \max_{1\leq i\leq N}||(x_i, 1)||_2$, since
$$
\begin{align}
||\theta_k||^2 &= (\theta_{k-1} + \alpha y_i(x_i, 1))^2\\
&= \theta_{k-1}^2 + 2\theta_{k-1} \alpha y_i(x_i, 1) + (\alpha y_i(x_i, 1))^2\\
&= \theta_{k-1}^2 + 2\theta_{k-1} \alpha y_i(x_i, 1) + \alpha^2 (x_i, 1)^2\\
&\leq \theta_{k-1}^2 + 2\theta_{k-1} \alpha y_i(x_i, 1) + \alpha^2R^2\\
&= \theta_{k-1}^2 + 2\alpha\theta_{k-1} (y_ix_i, x_i) + \alpha^2R^2\\
&\leq \theta_{k-1}^2 + \alpha^2R^2 \quad (since~ \theta_{k-1} (y_ix_i, x_i) ~is ~misclassified)
\end{align}
$$
By solving recurstion, we have $||\theta_k||^2  \leq k\alpha^2R^2\quad (B)$
From $(A)$ and $(B)$
$$
\begin{align}
k\alpha\gamma &\leq \theta_k \cdot \theta^* \quad (A)\\
&\leq |\theta_k||\theta^*|\quad (Cauchy)\\
&\leq \sqrt{k}\alpha R \quad(B)
\end{align}
$$
We can set $|\theta^*|$ to be 1, since we can adjust $w$ and $b$ propotionally.  
So we only need to itearate $k \leq (\frac{R}{r})^2$ times.

## Duality Form of Perceptron
In the previous version of algorithm, we see that 
$$
w = w + \alpha \sum_{x_i \in M} y_i x_i \\
b = b + \alpha \sum_{x_i \in M} y_i
$$
Then for a paticular point $(x_i, y_i)$, suppose it is used to update $w$ and $b$ for $n_i$ times, then $w$ and $b$ can be re-expressed as (consider updates all at once):
$$
w = \sum_{i = 1}^{N}\alpha n_i   y_i x_i \\
b =  \sum_{i = 1}^{N}\alpha n_i y_i
$$
$$
f(x) = sign(wx + b) = sign[(\sum_{i = 1}^{N}\alpha n_i   y_i x_i ) \cdot x + \sum_{i = 1}^{N}\alpha n_i y_i]
$$
> If $n_i$ is large, it means that data has been used for many times to update parameters, indicating that it is closer to hyperplane and contribute more.

## Duality Algorithm
* Initialize $w$ and $b$ as 0
*  Repeat until $l=0$
	* select data $(x_i, y_i)$
	* if $y_i(wx_i+b) = y_i[(\sum_{j=1}^{N}\alpha n_j y_jx_j)\cdot x_i + \sum_{j=1}^{N}\alpha n_j  y_j]\geq 0$
		* $n_i = n_i + 1$
	
## 
The reason why we want to have this algorithm is that, in part 
$(\sum_{j=1}^{N}\alpha n_j y_jx_j)* x_i$, we have a lot of inner products. If we can pre-compute them, it will be more efficient. And the way we use to calculate all inner products is by using $Gram$ Matrix, 
$$G=[x_{i}\ast x_j]_{N\times N} = \begin{bmatrix} \langle x_1, x_1 \rangle & \langle x_1, x_2 \rangle & \cdots & \langle x_1, x_n \rangle \\ \langle x_2, x_1 \rangle & \langle x_2, x_2 \rangle & \cdots & \langle x_2, x_n \rangle \\ \vdots & \vdots & \ddots & \vdots \\ \langle x_n, x_1 \rangle & \langle x_n, x_2 \rangle & \cdots & \langle x_n, x_n \rangle \\ \end{bmatrix}$$


# Decision Tree

## ID3 (Iterative Dichotomiser)
Based on information gain, we want to maximize the information gain at each step.
$$Information\space gain = Entropy(parent) - Entropy(children)$$
$$Entropy = -\sum_{i=1}^{n}p_i log_2 p_i$$
where $p_i$ is the probability of each class in the dataset.  
Drawback:
- It doesn't support missing values
- It doesn't support continuous values
- It doesn't have pruning, causing overfitting
- Not guranteed to be binary tree
  
## C4.5
Based on information gain ratio, we want to maximize the information gain ratio at each step.
$$gainRatio(D,A_{i})=\frac{gain(D,A_{i})}{-\sum_{s}^{j-1}{(\frac{|D_{j}|}{|D|}\ast log_{2}\frac{|D_{j}|}{|D|}})}$$  


Drawback:
- It's can branch out to more than 2 branches
- It only can be used for classification problems
- Logarithm function is expensive to compute

## CART
The Gini Index or Impurity measures the probability for a random instance being misclassified when chosen randomly. The lower the Gini Index, the better the lower the likelihood of misclassification.
### Gini Index
The formula for Gini Index:
$$Gini(D)=1-\sum_{i=1}^{n}p_{i}^{2}$$
since it's a binary tree, we can simplify it as:
$$Gini(D|A) = \frac{|D_1|}{|D|} \text{Gini}(D_1) + \frac{|D_2|}{|D|} \text{Gini}(D_2)$$  
> In english: weighted gini sum  
> 
Gini index happens to be the first order Taylor expansion of entropy since $xlog(x) \approx x(x-1)$ when $x$ is small.  
![Alt text](image-19.png)

### Dealing with continuous values
![All](image-20.png)  
Split points are achieved by sorting the values and finding the midpoints between adjacent values. Then it is converted into a discrete scenario again.

## Pre-Pruning
- Maximum Depth
- Minimum Number of Samples
- Setting a threshold for the minimum information gain
- Setting a threshold for the minimum Gini Index
drawback: it can be too conservative and underfit the data
## Post-Pruning
|            | CCP          | REP                | PEP                 | MEP                |
| ---------- | ------------ | ------------------ | ------------------- | ------------------ |
| Pruning Method | Bottom-Up | Bottom-Up | Top-Down | Bottom-Up |
| Computational Complexity | O(n^2) | O(n) | O(n) | O(n) |
| Error Estimation | Standard Error | Pruning Set Error | Continuity Correction | Probability Estimation |
| Requires Testing Set | No | Yes | No | No |

### Reduced Error Pruning (REP)
If pruning can reduce the error rate on the testing set, then we prune the tree.
### Pessimistic Error Pruning (PEP)
Error rate for a single node:
$$error(t) = \frac{e(t) + 0.5}{n(t)}$$
> +0.5 is the continuity correction
Error rate for a subtree:  

Error rate for a subtree:
$$Error(T) = \frac{ \sum_{i=1}^{L} e(i) + 0.5L }{ N(T)  }$$
> N(T) is the number of samples in the subtree
> L is the number of leaves in the subtree

Expected Error rate for a subtree:
$$E(T) = N(T) * Error(T)$$
Standard Deviation of the error rate for a subtree:
$$\sigma(T) = \sqrt{N(T) * Error(T) * (1 - Error(T))}$$
> We treat the error as a binomial distribution, so expectation is $np$ and variance is $np(1-p)$
Hence, the most pessimistic error rate for a subtree is:
$$E(T) + \sigma(T)$$  
If after pruning the tree, the error rate is lower than the most pessimistic error rate, then we prune the tree.
### Minimum Error Pruning (MEP)
At node T, the probability of being classified as class k is:
$$Error(T)= 1- \frac{n_k(T)}{N(T)}$$
$$=1 - \frac{n_k(T) + 1}{N(T) + K} \quad \text{(Laplace Smoothing)}$$
$$=\frac {N(T)-n_k(T)+K-1}{N(T)+K}$$
> $n_k(T)$ is the number of samples at node T classified as class k  
> $N(T)$ is the number of samples at node T  
> K is the total number of classes  
> The reason why it is not $1- \frac{n_k(T)}{N(T)}$ is that we use Laplace Smoothing to avoid the case where $n_k(T) = 0$. We can think of it as it gives every class an extra sample to avoid 0;

Before pruning, the error is:
$$Error(T_{before}) = \sum_{i=1}^{L} w_i \cdot Error(Leaf_i) = \sum_{i=1}^{l} \frac{N(Leaf_i)}{N(T)} \cdot Error(Leaf_i)
$$
After pruning, the error is:
$$Error(T_{after})= \frac {N(T)-n_k(T)+K-1}{N(T)+K}$$

If $Error(T_{after}) < Error(T_{before})$, then we prune the tree.


### Cost-Complexity Pruning (CCP)
The reason why is it called CCP is that error here consists of two parts: cost and complexity.
$$C_\alpha(T) = C(T) + \alpha |T|$$
where  
$$C(T) = \sum_{i=1}^{L}w_i\cdot Error(Leaf_i) \tag{Error(Leaf) is the misclassing rate}$$
If $C_\alpha(T_{after}) ≤ C_\alpha(T_{before})$, then we prune the tree.

# Bayes Classifier

## Bayes Theorem
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$
more generally, we have

$$P(Y = c_i | X = x) = \frac{P(X = x | Y = c_i) \cdot P(Y = c_i)}{P(X = x)} =\frac{P(X = x | Y = c_i) \cdot P(Y = c_i)}{\sum_{i=1}^{k} P(X = x | Y = c_i) \cdot P(Y = c_i)} $$
where $c_i$ is the ith class, $x$ is the input data,which has n features (i.e. $x$ = $(x_1, x_2, ..., x_n)$). $P(Y = c_i | X = x)$ is the probability of the input data being classified as class $c_i$.

## Naive Bayes Theorem
When the $n$ features from $x$ are independent, we have
$$P(X = x | Y = c_i) = \prod_{j=1}^{n} P(X^{(j)} = x^{(j)} | Y = c_i)$$
where $X^{(j)}$ is the jth feature of $X$.
By substituting the above formula into the Bayes Theorem, we have:
$$P(Y = c_i | X = x) = \frac{\prod_{j=1}^{n} P(X^{(j)} = x^{(j)} | Y = c_i) \cdot P(Y = c_i)}{\sum_{i=1}^{k} P(Y = c_i)\prod_{j=1}^{n} P(X^{(j)} = x^{(j)} | Y = c_i)} $$
## Bayes Classifier
Given above formula, the actual idea of Bayes Classifier comes to our mind naturally, which is:
$$arg \max_{c_i} P(X=x|Y=c_i)\cdot P(Y=c_i)$$
When we are considering which class a certain data belongs to, all the probabilities will have the same denominator, so we can ignore it.
## Native Bayes Classifier
Same philosophy applies here:
$$arg \max_{c_k}P(Y = c_k) \prod_{j=1}^{n} P(X^{(j)} = x^{(j)} | Y = c_k) $$

# Linear Regression with one variable
Our hypothesis model:  
$$h_\theta(x) = \theta_0 + \theta_1x$$
the function to estimate our errors between real data and our model  

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
where $m$ is the number of training examples, $x^{(i)}$ is the ith training example, $y^{(i)}$ is the ith output, $h_\theta(x^{(i)})$ is the hypothesis of the ith training example.

## Gradient Descent  
The idea behind gradient descent is as follows: Initially, we randomly select a combination of parameters (θ₀, θ₁, ..., θₙ), calculate the cost function, and then look for the next combination of parameters that will cause the cost function to decrease the most. We continue to do this until we reach a local minimum because we haven't tried all possible parameter combinations, so we cannot be sure if the local minimum we find is the global minimum. Choosing different initial parameter combinations may lead to different local minima.


In gradient descent, as we approach a local minimum, the algorithm automatically takes smaller steps. This is because when we get close to a local minimum, it's evident that the derivative (gradient) equals zero at that point. Consequently, as we approach the local minimum, the gradient values naturally become smaller, leading the gradient descent to automatically adopt smaller step sizes. This is the essence of how gradient descent works. Therefore, there's actually no need for further decreasing the learning rate (alpha).  

## Apply Gradient Descent into linear regression model  
the key here is to find the partial derivative of model parameter  
$$\theta_0 = \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$$
$$\theta_1 = \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)$$
$$\frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$
$$\frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$
> We need to update $\theta_0$ and $\theta_1$ simultaneously  
> 
where $\alpha$ is the learning rate, $m$ is the number of training examples, $x^{(i)}$ is the ith training example, $y^{(i)}$ is the ith output, $h_\theta(x^{(i)})$ is the hypothesis of the ith training example.  
This method would sometimes also be referred to as "Batch Gradient Descent" because each step makes use of all training data.

# Matrix Review  
[CS2109S CheatSheet](https://coursemology3.s3.ap-southeast-1.amazonaws.com/uploads/attachments/e3/2d/be/e32dbead0b0c575ef71d120059c1741af17a3085cba0fb5eb6278da9568d8289.pdf?response-content-disposition=inline%3B%20filename%3D%22matrix_calculus_cheatsheet.pdf%22&X-Amz-Expires=600&X-Amz-Date=20230928T083209Z&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA2EQN7A45RM2X7VPX%2F20230928%2Fap-southeast-1%2Fs3%2Faws4_request&X-Amz-SignedHeaders=host&X-Amz-Signature=a2bdde670ac0db2119bc0d20f73ffbaa1253b613d99f5c6ec52ee301b0fd9c29)

# Linear Regression with multiple variables  
## New notations  
$x^{(i)}$ means $i$th data ($i$th row) from the training set.

$x_{j}^{i}$ means the jth feature in the ith data row  
Suppose we have data $x_{1}$, $x_{2}$, $x_{3}$, $x_{4}$, we would add a bias column $x_{0}$ to be 1, so the feature matrix would have a dimension of (m, n + 1)

## Gradient Descent for Multiple Variables
$h_θ (x)=θ^T X=θ_0+θ_1 x_1+θ_2 x_2+...+θ_n x_n$  
taking derivatives, we have  
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$
where $m$ is the number of training examples, $x^{(i)}$ is the ith training example, $y^{(i)}$ is the ith output, $h_\theta(x^{(i)})$ is the hypothesis of the ith training example.
> $\theta_0$ is not included in the summation because it is not multiplied by any feature.
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
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$
$$= \frac{1}{2m} (X\theta - y)^T (X\theta - y)$$
$$= \frac{1}{2m} (\theta^T X^T - y^T) (X\theta - y)$$
$$= \frac{1}{2m} (\theta^T X^T X\theta - \theta^T X^T y - y^T X\theta + y^T y)$$
so
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{2m} (2 X^T X\theta - 2 X^T y)$$
let it be 0, we have
$$\frac{\partial}{\partial \theta_j} J(\theta) = 0$$

$$X^T X\theta = X^T y$$

$$\theta = (X^T X)^{-1} X^T y$$







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

For non-linear, we could also include polynomials like this  $h_θ (x)=g(θ_0+θ_1 x_1+θ_2 x_2+θ_3 x_1^2+θ_4 x_2^2 )$  


## Cost Function
If we still plug our function into Square Error function, the ultimate cost function would not have the property of convexity, so we need to come up with a new convex function to make sure that local min is global min.  

$$\begin{equation}
Cost(h_\theta(x), x) = 
\begin{cases}
    {-log(h_\theta(x))} & \text{if } y = 1 \\
    {-log(1- h_\theta(x))} & \text{if } y = 0 \\
\end{cases}
\end{equation}$$


Simplify it as $Cost(h_θ (x),y)=-y×log(h_θ (x))-(1-y)×log(1-h_θ (x))$  
Our ultimate cost function is: 
$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} Cost(h_\theta(x^{(i)}), y^{(i)})$$
$$= \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)} log(h_\theta(x^{(i)}))-(1-y^{(i)}) log(1-h_\theta(x^{(i)}))]$$
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
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

 
it's the same as linear regression!!!  
Proof:
![Alt text](image-22.png) 
Then we can update $\theta_i$ accordingly.  
> In this case feature scaling is also helpful in reducing the number of iterations  

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
![Alt text](image-23.png)

We would use a method called one v.s. all  
![Alt text](image-24.png)
in this case we have three classifiers $h_θ^{(i)} (x)$. In order to make predictions, we will plug in x to each classifier and select the one with the highest probability $max h_θ^{(i)} (x)$

# Regularization  
## Underfit and Overfit

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
$h_θ (x)=θ_0+θ_1 x_1+θ_2 x_2^2+θ_3 x_3^3+θ_4 x_4^4$, the reason why we have overfit is because of those terms with high degree, we want to minimize their impact on our model. So we choose $\theta$ s.t. it can minimize:  
$$\frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + 1000 \cdot \theta_3^2 + 1000 \cdot \theta_4^2$$

to include punishment for high degree terms.  

In general, we will have: 
$$J(\theta) = \frac{1}{2m} [\sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2]$$
where $\lambda$ is called **Regularization Parameter**
where λ is called **Regularization Parameter**


## Gradient Descent and Normal Equation Using Regularization in Linear Regression

* Gradient Descent
> important: $θ_0$ would never be part of regularization, so all index numbers of $θ_j$ start from 1

$$\theta_0 = \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)}$$
$$\theta_j = \theta_j - \alpha [\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m} \theta_j]$$
$$= \theta_j(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

- Note that the difference is each time $θ_j$ is multiplied with a number smaller than 1
- For $θ_0$, we take λ as 0

* Normal Equation
$$\theta = (X^T X + \lambda \cdot L)^{-1} X^T y$$
where L is a matrix of size $(n + 1) * (n + 1)$ with 0 at the top left and 1s down the diagonal, 0 everywhere else.

Rigorous proof:  

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$
$$= \frac{1}{2m} (X\theta - y)^T (X\theta - y) + \frac{\lambda}{2m} \theta^T \theta$$
$$= \frac{1}{2m} (\theta^T X^T - y^T) (X\theta - y) + \frac{\lambda}{2m} \theta^T \theta$$
$$= \frac{1}{2m} (\theta^T X^T X\theta - \theta^T X^T y - y^T X\theta + y^T y) + \frac{\lambda}{2m} \theta^T \theta$$
hence the derivative is:
$$\frac{\partial}{\partial \theta} J(\theta) = \frac{1}{m} (X^T X\theta - X^T y) + \frac{\lambda}{m} \theta$$
Let it be 0, we have:
$$\frac{1}{m} (X^T X\theta - X^T y) + \frac{\lambda}{m} \theta = 0$$
$$(X^T X + \lambda \cdot L)\theta = X^T y$$
$$\theta = (X^T X + \lambda \cdot L)^{-1} X^T y$$
where L is a matrix of size $(n + 1) * (n + 1)$ with 0 at the top left and 1s down the diagonal, 0 everywhere else.

* If λ > 0, it is guaranteed to be invertible;  



## Gradient Descent Using Regularization in Logistic Regression
> important: $θ_0$ would never be part of regularization, so all index numbers of $θ_j$ start from 1

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)} log(h_\theta(x^{(i)}))-(1-y^{(i)}) log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

``` python
import numpy as np
def costReg(theta, X, y, lambda):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = lambda / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
```
We will have the following update models:  
$$\theta_0 = \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} ((h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)})$$
$$\theta_j = \theta_j - \alpha [\frac{1}{m} \sum_{i=1}^{m} ((h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}) + \frac{\lambda}{m} \theta_j]$$
$$= \theta_j(1 - \alpha \frac{\lambda}{m}) - \alpha \frac{1}{m} \sum_{i=1}^{m} ((h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)})$$
## Gradient Descent Using Regularization in Neural Networks 
$$J(w^{[1]}, b^{[1]}, ..., w^{[L]}, b^{[L]}) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m} \sum_{l=1}^{L} ||w^{[l]}||_F^2$$
where $||w^{[l]}||_F^2 = \sum_{i=1}^{n^{[l-1]}} \sum_{j=1}^{n^{[l]}} (w_{ij}^{[l]})^2$, basically it is the sum of all the square of all the weight parameters.

We need to include an extra term when updating.  
$$dw^{[l]} = \frac{\partial L}{\partial w^{[l]}} + \frac{\lambda}{m} w^{[l]}$$
hence, 
$$w^{[l]} = w^{[l]} - \alpha dw^{[l]}$$

## Dropout (In Neural Networks)
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


## Activation functions  

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
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} [y_k^{(i)} log((h_\theta(x^{(i)}))_k) + (1 - y_k^{(i)}) log(1 - (h_\theta(x^{(i)}))_k)] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} (\theta_{ji}^{(l)})^2$$
where $y_k^{(i)}$ indicates whether the label of the ith training example is k, and $h_\theta(x^{(i)})_k$ is the kth output.

It looks intimidating, but the idea remains the same, the only difference is that the final output is a vector of dimension **K**, $(h_θ(x))_i$ = i^th output in the result vector, the last summation term is just simply the sum of all the parameters (all weights) excluding the term where i = 0, which means it is multiplied with bias, so there's no need to take them into account. (If you do nothing will happen)  
$θ_{ji}^{(l)}$ here represents a parameter or weight in a neural network, specifically for layer "l" connecting the "i-th" neuron in layer "l" to the "j-th" neuron in layer "l+1" mathematically, $θ_{ji}^{(l)}$ is weight connecting $a_i^{l}$ and $a_j^{l+1}$  
A concrete example: $θ_{10}^{(2)}$ is the weight connecting $a_0^{(2)}$ at layer 2 with $a_1^{(3)}$

## Backpropagation Algorithm
The idea is to compute the error term of each layer and then use it to update the weights.  
Our goal always remain the same: we have a set of parameters waiting to be optimized, and we also have cost function. The essence of gradient descent is to find the partial derivative of cost function to all the parameters.  

Details and mathematical proof can be found at [here](https://github.com/WangCheng0116/ML/blob/main/Challenges/Back%20Propogation.md).

## Randomize Initialization  
If we still use 0 to initialize, it is easy to prove the whole updating procedure will be completely symmetric.  
so what we will do is initialize $w^{[i]}$ with a random number from [0, 1] * 0.01, the reason is we want it to have values close to 0 where the slope will be relatively large.  
for $b^{[i]}$, it won't affect symmetry so we could initialize it to be 0


# Deep Neural Network
## Working Flow Chart  
![Alt text](image-25.png)
By caching the value of $z^{[i]}$, we could make bp easier.  
![Alt text](image-26.png)
![Alt text](image-27.png)
![Alt text](image-28.png)
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
![Alt text](image-29.png)
![Alt text](image-30.png)
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

For each mini-batch, compute dW and db. At the $t$-th iteration:

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

**Related Paper:** [LeCun et al., 1998. Gradient-based learning applied to document recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=726791&tag=1). 

---

## AlexNet

![AlexNet](https://raw.githubusercontent.com/bighuang624/Andrew-Ng-Deep-Learning-notes/master/docs/Convolutional_Neural_Networks/AlexNet.png)

**Features:**

- AlexNet is similar to LeNet-5 but more complex, containing about 60 million parameters. Additionally, AlexNet uses the ReLU function.
- When used to train image and data sets, AlexNet can handle very similar basic building blocks, often comprising a large number of hidden units or data.

**Related Paper:** [Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

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

# Expectation Maximization (EM) Algorithm
## Intuitive Explanation
Suppose we want collect the data about the distribution of heights in NUS. We successfully gather the height data from 100 boys and 100 girls, but someone removed all labels by accident, which means we don't know whether a data come from boys or girls.
Now we face two issues:
* Suppose we know the labels again, then we can use [MLE](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#) to estimate the distribution of boys and girls respectively.
* Suppose we know for example, boys' heights ~ $N(\mu_1 = 172, \sigma^2_1=5^2)$, girls' height ~$N(\mu_2 = 162, \sigma^2_2=5^2)$.Then when given a data, we know which class is more likely to be the source of this data. For example, 180 cm is more likely from boys.  

But the thing is we know neither, so it becomes a problem of "Which came first, the chicken or the egg?". Well the answer is we can use EM algorithm to solve this problem.

## Intuitive Algorithm
1. **Initialization:** We start by setting the initial distribution parameters for male and female heights. For instance, the height distribution for males is assumed to be $N(\mu_1 = 172, \sigma^2_1 = 5^2)$, and for females, it's $N(\mu_2 = 162, \sigma^2_2 = 5^2)$. These initial values may not be very accurate.

2. **Expectation (E-Step):** We calculate the likelihood of each individual belonging to the male or female group based on their height. For example, a person with a height of 180 is highly likely to belong to the male group.

3. **Maximization (M-Step):** After roughly assigning the 200 people to either the male or female group in the E-Step, we estimate the height distribution parameters for males and females separately using the Maximum Likelihood Estimation (MLE) method.

4. **Iteration:** We update the parameters of these two distributions. As a result, the probability of each student belonging to either males or females changes again, requiring us to adjust the E-Step once more.

5. **Convergence:** We repeat these E-Step and M-Step iterations iteratively until the parameters no longer change significantly, or until specific termination conditions are met.

> In this case, hidden variables are labels, namely boys and girls. And the observed variables are heights.

## Formal Proof
### Likelihood Function
> What is the probablity that I happen to choose $n$ samples?  

Every sample is drawn from $p(x|θ)$ independently, so the probability of choosing them altogether at once is:
$$ L(\theta) = \prod_{i=1}^n p(x^{(i)}|θ)$$
### Log Likelihood Function
By logging the $L(\theta)$, we can get the log likelihood function:
$$H(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(x^{(i)}|θ)$$
> The reason why we choose log likelihood function is that it is easier to calculate and avoid the situation where the product of many small numbers is too small to be accurate in computer.
### Jensen Inequality
If $f(x)$ is a convex function, then
$$E[f(x)] \geq f(E[x])$$
specially, if $f(x)$ is strictly convex, then 
$E[f(x)] = f(E[x])$ if and only if  $p(x = E(x)) = 1$, i.e. random variable $x$ is a constant.
> If $f(x)$ is concave, flip the inequality sign.

### EM 
> Following proof doesn't use vectorization.

For $m$ independent data $x=(x^{(1)},x^{(2)},...,x^{(m)})$, and corrsponding hidden variables $z=(z^{(1)},z^{(2)},...z^{(m)})$. Then $(x,z)$ is full data.  

We want to find $\theta$ and $z$ s.t. it can maximize the likelihood function $L(\theta, z)$, i.e
$$
\theta, z=
\arg \max _{\theta, z} \sum_{i=1}^m \log \sum_{z^{(i)}} P\left(x^{(i)}, z^{(i)} \mid \theta\right) 
$$
We can see here that the summation is inside the log function. We want to extract it out:  
$$
\begin{aligned}
\sum_{i=1}^m \log \sum_{z^{(i)}} P\left(x^{(i)}, z^{(i)} \mid \theta\right) & =\sum_{i=1}^m \log \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_i\left(z^{(i)}\right)} \\
& \geq \sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_i\left(z^{(i)}\right)} \quad \text{(log(E[x]) >= E[log(x)] (because log(x) is concave))}
\end{aligned}
$$
where $Q_i(z^{(i)})$ is a distribution over $z^{(i)}$, it satisfies $\sum_{z^{(i)}} Q_i\left(z^{(i)}\right)=1$ and $1 \geq Q_i\left(z^{(i)}\right) \geq 0$.
> This is also called Expectation Step (E-Step), because $E\left(\log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_i\left(z^{(i)}\right)}\right)=\sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_i\left(z^{(i)}\right)}$

Suppose $\theta$ has been fixed, then $logL(\theta)$ is determined by two values, $Q_i(z^{(i)})$ and $P\left(x^{(i)}, z^{(i)} \mid \theta\right)$. We want to maximize the lower bound of $logL(\theta)$, so we need to make the inequality hold.

In order to hold the inequality, we need to let 
$$
\frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_i\left(z^{(i)}\right)}=c
$$
$$\Leftrightarrow P\left(x^{(i)}, z^{(i)} \mid \theta\right)=cQ_i\left(z^{(i)}\right) \quad \text{(1)}$$
$$
\Leftrightarrow
\sum_z P\left(x^{(i)}, z^{(i)} \mid \theta\right)=c \sum_z Q_i\left(z^{(i)}\right) \quad \text{(sum over z)}
$$
$$
\Leftrightarrow
\sum_z P\left(x^{(i)}, z^{(i)} \mid \theta\right) = c
\quad \text{(2)}
$$
By $(1)$ and $(2)$,
$$
\begin{gathered}

Q_i\left(z^{(i)}\right)=\frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{c}=\frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{\sum_z P\left(x^{(i)}, z^{(i)} \mid \theta\right)}=\frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{P\left(x^{(i)} \mid \theta\right)}=P\left(z^{(i)} \mid x^{(i)}, \theta\right)
\end{gathered}
$$
since 
* Marginal Probability$: \space P\left(x^{(i)} \mid \theta\right)=\sum_z P\left(x^{(i)}, z^{(i)} \mid \theta\right)$
* Conditinonal Probability$: \space \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{P\left(x^{(i)} \mid \theta\right)}=P\left(z^{(i)} \mid x^{(i)}, \theta\right)$  

> We can interpret $P\left(z^{(i)} \mid x^{(i)}, \theta\right)$ as the probability of $z^{(i)}$ given $x^{(i)}$ and $\theta$.


Hence we have successfully found the $Q_i(z^{(i)}) = P\left(z^{(i)} \mid x^{(i)}, \theta\right)$ that maximizes the lower bound of $logL(\theta)$.
What we are left to do is to maximize:  
$$
\begin{aligned}
\sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta\right)}{Q_i\left(z^{(i)}\right)} &=\sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log P\left(x^{(i)}, z^{(i)} \mid \theta\right)-\sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log Q_i\left(z^{(i)}\right) \\
&=\sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log P\left(x^{(i)}, z^{(i)} \mid \theta\right)+H(Q_i)
\end{aligned}
$$
$$
\approx \sum_{i=1}^m \sum_{z^{(i)}} P\left(z^{(i)} \mid x^{(i)}, \theta\right) \log P\left(x^{(i)}, z^{(i)} \mid \theta\right) \quad \text{(by leaving out the constant)}
$$
where $H(Q_i)$ is the entropy of $Q_i(z^{(i)})$ and it is a constant.
## EM Algorithm
**Input:**
Observation data $x = (x^{(1)}, x^{(2)}, \ldots, x^{(m)})$,
joint distribution $p(x, z \, | \, \theta)$,
conditional distribution $p(z \, | \, x, \theta)$,
maximum iteration count $J$.

1. Randomly initialize the initial values of model parameters $\theta$ as $\theta^{(0)}$.

2. For each $j$ from 1 to $J$:  
    **E-step:**  
    Calculate the conditional probability expectations of the joint distribution:  
    $$
    Q_i\left(z^{(i)}\right)=P\left(z^{(i)} \mid x^{(i)}, \theta\right)
    $$
    > After substituting $\theta$, $Q_i\left(z^{(i)}\right)$ is a constant.

    **M-step:**  
    Maximize the likelihood function with respect to $\theta$:
    $$
    \theta=\arg \max _{\theta} \sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log {P\left(x^{(i)}, z^{(i)} \mid \theta\right)}
    $$
3. Repeat step 2 until convergence.
> [Video of EM Algorithm](https://www.bilibili.com/video/BV19a411N72s/?spm_id_from=333.337.search-card.all.click&vd_source=43fae35e0d515715cd36645ea2e6e547)
## Why EM Converges?
$$
H(\theta^{(t+1)}) \geq \sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta^{(t+1)}\right)}{Q_i\left(z^{(i)}\right)} 
$$

$$
\geq \sum_{i=1}^m \sum_{z^{(i)}} Q_i\left(z^{(i)}\right) \log \frac{P\left(x^{(i)}, z^{(i)} \mid \theta^{(t)}\right)}{Q_i\left(z^{(i)}\right)}
\qquad \text{(by why we get $\theta^{(i + 1)}$)}
$$
$$
=H(\theta^{(t)})
$$
> I feel sus abt the proof. TBC later.
# Gaussian Mixture Model (GMM)
> GMM is a special case of EM algorithm.
## Notation
* Gaussian Mixtures: 
$$p(x) = \sum_{i=1}^k \pi_i N(x|\mu_i, \Sigma_i)$$
* $\pi_i$ is the weight of the $i$-th Gaussian, and $\sum_{i=1}^k \pi_i = 1$.
* 
$$\gamma(i, k) 
= \frac{\pi_i N(x|\mu_i, \Sigma_i)}{\sum_{j=1}^k \pi_j N(x|\mu_j, \Sigma_j)}$$
 is the probability of $i$-th data belonging to the $k$-th Gaussian distribution.
> $\gamma(i, k)$ is simply a handy notation to extract all the data out which are from the $k$-th Gaussian distribution. Don't be confused by the notation.
## Algorithm
- **E-Step:** Calculate $\gamma(i, k)$
$$\gamma(i, k) 
= \frac{\pi_i N(x|\mu_i, \Sigma_i)}{\sum_{j=1}^k \pi_j N(x|\mu_j, \Sigma_j)}$$
> The E-step computes these probabilities using the current estimates of the model's parameters. 
- **M-Step:** Update parameters
$$  
\left\{
\begin{array}{l}
N_k = \sum_{i=1}^{N} \gamma(i, k) \\
\\  
\\
\mu_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(i, k) x_i \\
\\ 
\\
\sum_k = \frac{1}{N_k} \sum_{i=1}^{N} \gamma(i, k) (x_i - \mu_k)(x_i - \mu_k)^T \\
\\  
\\
\pi_k = \frac{N_k}{N} = \frac{1}{N} \sum_{i=1}^{n} \gamma(i, k)
\end{array}
\right.
$$


## Intuitive Example
Again, back to the height example.  

Suppose we have the following data:
$$
\begin{array}{l}
x^{(1)} = 180 \\
x^{(2)} = 170 \\
x^{(3)} = 160 \\
x^{(4)} = 155 \\
\end{array}
$$

We initialize two Gaussian distributions (for males and females respectively) as follows:
$$
\mu_1 = 170 \space \mu_2 = 160$$
$$
\sigma_1 = 0.5 \space \sigma_2 = 0.7
$$
$$
\pi_1 = 0.5 \space \pi_2 = 0.5
$$
Then 
$$
\begin{array}{l}
\gamma(1, 1) = 0.8 \quad \gamma(1, 2) = 0.2 \\
\gamma(2, 1) = 0.6 \quad \gamma(2, 2) = 0.4 \\
\gamma(3, 1) = 0.4 \quad \gamma(3, 2) = 0.6 \\
\gamma(4, 1) = 0.2 \quad \gamma(4, 2) = 0.8 \\
\end{array}
$$
> eg. $\gamma(3, 1)$ means the probability of 160cm person is from Male distribution.

For $k=1$:
$$
\begin{array}{l}
N_1 = 0.8 + 0.6 + 0.4 + 0.2 = 2 
\\
\\
\mu_1 = \frac{1}{2} (180 \times 0.8 + 170 \times 0.6 + 160 \times 0.4 + 155 \times 0.2) = 171.5 
\\
\\
\sigma_1 = \frac{1}{2} (0.8 \times (180 - 171.5)^2 + 0.6 \times (170 - 171.5)^2 + 0.4 \times (160 - 171.5)^2 + 0.2 \times (155 - 171.5)^2) = 25.5 
\\
\\
\pi_1 = \frac{2}{4} = 0.5
\end{array}
$$

Same applies to $k=2$.

Then we update the parameters and calculate $\gamma$ again. We repeat this process until convergence.

## Implementation
Credits to [this](https://github.com/Ransaka/GMM-from-scratch/blob/master/GMM%20from%20scratch.ipynb), which is a very good implementation of GMM from scratch.
## Math Proof
(to be added)
 



# Support Vector Machine
## Proof
### Problem Formulation
Suppose we have traning data $$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)} $$
where $y_i\in \left\{ +1,-1 \right\} , i=1,2,...N$

We need to find a hyperplane which can classify data perfectly, i.e.,
$$ f(x) = \begin{cases} \omega^Tx_i + \gamma > 0 & \text{if } y_i = 1 \\ \omega^Tx_i + \gamma < 0 & \text{if } y_i = -1 \end{cases} $$
By setting the distance between hyperplane and support vector to be $d$, then 
$$ f(x) = \begin{cases} \frac{\omega^Tx_i + \gamma}{||w||} \geq d & \text{if } y_i = 1 \\ \frac{\omega^Tx_i + \gamma}{||w||} \leq d & \text{if } y_i = -1 \end{cases} $$
Divide $d$ on both sides, and 
$$ f(x) = \begin{cases} \omega_d^Tx_i + \gamma_d \geq 1 & \text{if } y_i = 1 \\ \omega_d^Tx_i + \gamma_d \leq -1 & \text{if } y_i = -1 \end{cases} $$
where,
$$\omega_{d}=\frac{\omega}{||\omega||d},\ \ \gamma_{d}=\frac{\gamma}{||\omega||d}$$
Since they still represent lines, we can simplify them as,
$$ f(x) = \begin{cases} \omega^Tx_i + \gamma \geq  1 & \text{if } y_i = 1 \\ \omega^Tx_i + \gamma \leq -1 & \text{if } y_i = -1 \end{cases} $$
$$\Leftrightarrow y_{i}(\omega^{T}x_{i}+\gamma)\geq1\ ,for \space \forall x_{i}$$
Distance $d =$ distance between support vectors and hyperplane,
$$d = \frac{\omega^{T}x_{i}+\gamma}{||w||} = \frac{1}{||w||} $$
since points on support vectors satisfy $\omega^{T}x_{i}+\gamma = 1$

$$max \space d \\
\Leftrightarrow min \space\frac{1}{2}||w||^2\\
s.t.  \space y_{i}(\omega^{T}x_{i}+\gamma)\geq1\ ,\forall x_{i}$$

### Optimization Problem
Rewrite it as a KKT problem
$$
min \space\frac{1}{2}||w||^2\\
s.t.  \space h_i(x) = 1- y_{i}(\omega^{T}x_{i}+b)\leq0\ ,i = 1, 2,...,N$$
Constructing the Lagrangian Function
$$
{\mathcal{L}}(w,b,\alpha)={\frac{1}{2}}\|w\|^{2}+\sum_{i=1}^{n}\alpha_{i}h_i(x)\\
={\frac{1}{2}}\|w\|^{2}-\sum_{i=1}^{n}\alpha_{i}\left(y_{i}(w^{\mathrm{T}}x_{i}+b)-1\right)$$
Orignial question can be converted to
$$\operatorname*{min}_{w,b} \frac{1}{2}||w||^2=\operatorname*{min}_{w,b}\operatorname*{max}_{\alpha_i\geq 0}{\mathcal{L}}(w,b,\alpha)
$$
## Duality Conversion
We want to show the above function satisfies two properties
* Convexity 
* [Slater's Condition](https://en.wikipedia.org/wiki/Slater%27s_condition#Formulation)
> In English, Slater's means that there exist a feasible point which let all inequality constrain cannot hold equality, i.e. $h_i(x) <0$

If so we can use Strong Langrangian Duality. 
* Convexity: it holds true for $\frac{1}{2}||w||^2$ and $h_i(x)$
* Slater's: it holds true since all inequality constrain are affine

Strong duality holds. We can now solve this instead,
$$\operatorname*{max}_{\alpha_i\geq 0}\operatorname*{min}_{w, b}{\mathcal{L}}(w,b,\alpha)$$
We first fix the value of all $\alpha_i$, since we want to minimize $\mathcal{L}$, we should take partial derivatives w.r.t $w$ and $b$, 

$${\frac{\partial{\mathcal{L}}}{\partial w}}=0\Rightarrow w=\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}$$
$${\frac{\partial{\mathcal{L}}}{\partial b}}=0\Rightarrow\sum_{i=1}^{n}\alpha_{i}y_{i}=0$$
By plugging them back in to $\mathcal{L}$, we have
$$\begin{align*}
min \space \mathcal{L}(w,b,\alpha) &= \frac{1}{2}\|w\|^2 - \sum_{i=1}^{n}\alpha_{i}\left[y_{i}(w^T x_i + b) - 1\right] \\
&= \frac{1}{2}w^T w - w^T \sum_{i=1}^{n}\alpha_{i}y_{i}x_{i} - b\sum_{i=1}^{n}\alpha_{i}y_{i} + \sum_{i=1}^{n}\alpha_{i} \\
&={\frac{1}{2}}w^{\mathrm{{T}}}\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}-w^{\mathrm{{T}}}\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}-b\cdot0+\sum_{i=1}^{n}\alpha_{i} \\
&=\sum_{i=1}^{n}\alpha_{i}-{\frac{1}{2}}\left(\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i}\right)^{\mathrm{T}}\sum_{i=1}^{n}\alpha_{i}y_{i}x_{i} \\
&={\sum_{i=1}^{n}}\alpha_{i}-{\frac{1}{2}}\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{\mathrm{{T}}}x_{j}
\end{align*}
$$
This is $min$ part  inside of $max \space min$, so the $max$ is 
$$\operatorname*{max}_{\alpha}\sum_{i=1}^{n}\alpha_{i}-\frac12\sum_{i,j=1}^{n}\alpha_{i}\alpha_{j}y_{i}y_{j}x_{i}^{\mathrm{T}}x_{j} \\
s.t. \alpha_i \geq 0, i = 1, 2, ..., n\\
\sum_{i=1}^{n}\alpha_iy_i=0$$

## Algorithm (HardMargin)
1. Solve
$$
\operatorname*{max}_{\alpha} \sum_{i=1}^{N} \alpha_i-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) 
$$
subject to
$$\sum_{i=1}^{N} \alpha_i y_i = 0 \\\alpha_{i}\geqslant 0$$
to get optimal solution $\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},\cdot\cdot\cdot,\alpha_{N}^{*})^{\mathrm{T}}$
3. 
	$\omega^{*}=\sum_{i-1}^{N}\alpha_{i}^{*}y_{i}x_{i}$, and choose any $j$ s.t. $\alpha_j > 0$ to get $b^{*}=y_{j}-\sum_{i=1}^{N}y_{i}\alpha_{i}^{*}(x_{i}\ast x_{j})$	
4. Hyperplane is $w^{T}x+b^{*}=0$
## SoftMargin
For the majority of cases, there are some points fall in between two support vectors inevitably, i.e. Not all points can satisfy 
$$y_{i}(w^T x_i + b) \geq 1$$
To tackle this issue, we introduce another variable for the sake of slackness. 
$$y_{i}(w^T x_{i}+b)+ \xi_{i} \geq 1$$
where $\xi_i \geq 0$ 

So the updated question is,
$$\operatorname*{min}_{w,b,\xi}~~{\frac{1}{2}}\|w\|^{2}+C\sum_{i=1}^{N}\xi_{i} \\
s.t.  \space 1 - y_{i}(w^T x_{i}+b)  - \xi_{i} \leq 0 ~~ \text{(a)}\\
 -\xi_{i} \leq 0 ~~ \text{(b)}$$

Lagrangian Function
$${\mathcal{L}}(w,b,\alpha, \mu, \xi) = {\frac{1}{2}}\|w\|^{2}  +C\sum_{i=1}^{N}\xi_{i} +\sum_{i=1}^{N}\alpha_{i}(1 - y_{i}(w^T x_{i}+b)  - \xi_{i}) + \sum_{i=1}^{N}\mu_i(-\xi_i)$$
Want to find
$$\operatorname*{min}_{w,b,\xi} \operatorname*{max}_{\alpha, \mu} \mathcal{L}$$
which is equaivalent with,
$$\operatorname*{max}_{\alpha, \mu} \operatorname*{min}_{w,b,\xi}\mathcal{L}$$

By taking derivative w.r.t $w$, $b$ and $\xi$, we have 
$$\begin{array}{c}{{w=\sum_{i=1}^{N}\alpha_{i}y_{i}x_{i}}}\\ 
\sum_{i=1}^{N}\alpha_iy_i =0\\
{{C-\alpha_{i}-\mu_{i}=0}}\end{array}$$

By plugging in, the dual problem now is

$$
\operatorname*{max}_{\alpha} \sum_{i=1}^{N} \alpha_i-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) 
$$
subject to

$$
\sum_{i=1}^{N} \alpha_i y_i = 0 \\
C - \alpha_i - \mu_i = 0 \\
\alpha_i \geq 0, \quad \text{for } i = 1,2,\ldots,N \\
\mu_i \geq 0, \quad \text{for } i = 1,2,\ldots,N \\
$$
constrain can be simplified as
$$\sum_{i=1}^{N} \alpha_i y_i = 0 \\0\leqslant\alpha_{i}\leqslant C$$

## Algorithm (SoftMargin)
1. Select punishment coeffient $c$
2. Solve
$$
\operatorname*{max}_{\alpha} \sum_{i=1}^{N} \alpha_i-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) 
$$
subject to
$$\sum_{i=1}^{N} \alpha_i y_i = 0 \\0\leqslant\alpha_{i}\leqslant C$$
to get optimal solution $\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},\cdot\cdot\cdot,\alpha_{N}^{*})^{\mathrm{T}}$
3. 
	$\omega^{*}=\sum_{i-1}^{N}\alpha_{i}^{*}y_{i}x_{i}$, and choose any $j$ s.t. $0 < \alpha_j < C$ to get $b^{*}=y_{j}-\sum_{i=1}^{N}y_{i}\alpha_{i}^{*}(x_{i}\ast x_{j})$	
4. Hyperplane is $w^{T}x+b^{*}=0$

## Another Perspective: HingeLoss
First, we define HingeLoss Function as
$$L(y(w\ast x+b))=[1-y(w\ast x+b)]_{+}$$
where
$$[z]_+ = \begin{cases}
    z, & \text{if } z > 0 \\
    0, & \text{otherwise}
\end{cases}
$$
So we want to minimize the cost function (with regularization)
$$\operatorname*{min}_{w, b} \sum_{i=1}^{N}\left[1-y_{i}(w*x_{i}+b)\right]_{+}+\lambda\vert\vert w\vert\vert^{2} ~~~\text{(1)}$$

Now we want to show this is equivalent with 
$$\operatorname*{min}_{w,b,\xi}~~{\frac{1}{2}}\|w\|^{2}+C\sum_{i=1}^{N}\xi_{i} ~~~ \text{(2)}$$

Proof:
Let
$$[1-y_{i}(w^T x_{i}+b)]_{+}=\xi_{i} \geq 0 ~~~\text{(trivial)}$$
and it satisfies
$$y_{i}(w^T x_{i}+b) \geq 1-\xi_{i} ~~~ \text{(divide into cases)}$$

Hence, if we take $\lambda=\frac{1}{2C}$, $(1)$ and $(2)$ are the same because they have same objective function and same constrain.


## SMO (Sequential Minimal Optimization)
$$
\operatorname*{min}_{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)  -\sum_{i=1}^{N} \alpha_i
$$
subject to
$$\sum_{i=1}^{N} \alpha_i y_i = 0 \\0\leqslant\alpha_{i}\leqslant C$$
We want to get optimal solution $\alpha^{*}=(\alpha_{1}^{*},\alpha_{2}^{*},\cdot\cdot\cdot,\alpha_{N}^{*})^{\mathrm{T}}$
We choose to treat $\alpha_1$ and $\alpha_2$ like variable while others are constants ($\alpha_i, i=3,...,N$).
Let 
$$
L(\alpha_1, \alpha_2) = \frac{1}{2}[\alpha_1 y_1 \alpha_1 y_1 x_1^T \cdot x_1 + 2 \alpha_1 y_1 \alpha_2 y_2 x_1^T \cdot x_2 + 2 \sum_{j=3}^{N} \alpha_1 y_1 \alpha_j y_j x_1^T \cdot x_j + \alpha_2 y_2 \alpha_2 y_2 x_2^T \cdot x_2 + 2 \sum_{j=3}^{N} \alpha_2 y_2 \alpha_j y_j x_2^T \cdot x_j + \sum_{i=3}^{N} \sum_{j=3}^{N} \alpha_i y_i \alpha_j y_j x_i^T \cdot x_j] - [\alpha_1 + \alpha_2 + \sum_{j=3}^{N} \alpha_j]
$$
Let $x_{i}^{T}\cdot x_{j}=K_{ij}$, then
$$
L(\alpha_1, \alpha_2) = \frac{1}{2}[\alpha_1 y_1 \alpha_1 y_1 K_{11} + 2 \alpha_1 y_1 \alpha_2 y_2 K_{12} + 2 \sum_{j=3}^{N}{\alpha_1 y_1 \alpha_j y_j K_{1j}} + \alpha_2 y_2 \alpha_2 y_2 K_{22} + 2 \sum_{j=3}^{N}{\alpha_2 y_2 \alpha_j y_j K_{2j} + \sum_{i=3}^{N}{\sum_{j=3}^{N}{\alpha_i y_i \alpha_j y_j K_{ij}}}] - [\alpha_1 + \alpha_2 + \sum_{j=3}^{N}{\alpha_j}]}
$$
Since $\sum_{i=3}^{N} \sum_{j=3}^{N} \alpha_i y_i \alpha_j y_j K_{ij}~~$and$~~\sum_{j=3}^{N}{\alpha_{j}}$ are constants, we can discard them without affecting the results of differentiation. We have
$$
L(\alpha_1, \alpha_2) = \frac{1}{2}[\alpha_1^2 K_{11} + 2 \alpha_1 y_1 \alpha_2 y_2 K_{12} + 2 \sum_{j=3}^{N} \alpha_1 y_1 \alpha_j y_j K_{1j} + \alpha_2^2 K_{22} + 2 \sum_{j=3}^{N} \alpha_2 y_2 \alpha_j y_j K_{2j}] - [\alpha_1 + \alpha_2]
$$

Let $\sum_{i=3}^{N} \alpha_i y_i = -C$, then $\alpha_1 y_1 + \alpha_2 y_2 - C = 0$
Substituting $\alpha_1 = y_1(C - \alpha_2 y_2)$:

$$L(\alpha_2) = \frac{1}{2}[(C-\alpha_2y_2)^2K_{11} + 2(C -\alpha_2y_2 )\alpha_2y_2K_{12} + 2\sum_{j=3}^{N}(C -\alpha_2y_2 )\alpha_jy_jK_{1j} + \alpha_2^2K_{22} +2\sum_{j=3}^{N}\alpha_2y_2\alpha_jy_jK_{2j} ] - [y_1(C- \alpha_2y_2) + \alpha_2]$$

By letting $\frac{\sigma L}{\sigma \alpha_{2}}= 0$, we have
$$\begin{align*}
\alpha_2(K_{11} + K_{22} - 2K_{12}) &= 1 - y_1 y_2 + C y_2 K_{11} - Cy_2 K_{12} + \sum_{i=3}^{N} y_2 \alpha_i y_i K_{1i} - \sum_{i=3}^{N} y_2 \alpha_j y_j K_{2j}  \quad ~~~~~ (1 = y_2^2 )\\
&= y_2 [y_2 - y_1 + C K_{11} - C K_{12} + \sum_{i=3}^{N} \alpha_i y_i K_{1i} - \sum_{i=3}^{N} \alpha_i y_i K_{2i}] \quad (\text{Equation 1})
\end{align*}
$$

Given that $w = \sum_{i=1}^{N}{\alpha_{i}y_{i}x_{i}}$ and $f(x) = w^{T}x + b$, it follows that $f(x) = \sum_{i=1}^{N}{\alpha_{i}y_{i}x_{i}^{T}}x + b$.
So, $f(x_1) - \alpha_1 y_1 K_{11} - \alpha_2 y_2 K_{12} - b = \sum_{i=3}^{N}{\alpha_i y_i K_{1i}}$ (Equation 2)
$f(x_2) - \alpha_1 y_1 K_{12} - \alpha_2 y_2 K_{22} - b = \sum_{i=3}^{N}{\alpha_i y_i K_{2i}}$ (Equation 3)
By substituting $Equation ~2$ and $Equation ~3$ into $Equation ~1$, and $\alpha_{1}^{old}y_{1}+\alpha_{2}^{old}y_{2}=C$, we have
$$\alpha_{2}^{new,unc}(K_{11}+K_{22}-2K_{12}) = y_{2}[(f(x_{1})-y_{1}) - (f(x_{2})-y_{2}) + \alpha_{2}^{old}y_{2}(K_{11}+K_{22}-2K_{12})]$$

Let $E_{1} = f(x_{1}) - y_{1}$ and $E_{2} = f(x_{2}) - y_{2}$, and $\eta = K_{11} + K_{22} - 2K_{12}$, so 
$$\alpha_{2}^{new,unc} = \alpha_{2}^{old} + \frac{y_{2}(E_{1} - E_{2})}{\eta}$$
$$\alpha_1^{new, unc} = y_1(C - \alpha_2^{new, unc}y_2)$$

The reason why we have a special notation $\alpha_{2}^{new,unc}$ is that we are just letting partial derivative to be $0$ without considering whether $\alpha_2$ can meet the contrain $\alpha_1 y_1 + \alpha_2 y_2 - C = 0$ or not.
> ERRETA: I shouldn't use C in $\alpha_1 y_1 + \alpha_2 y_2 - C = 0$ because it has been used in the contrain where $0\leqslant\alpha_{i}\leqslant C$.  


Let $L$ and $H$ be the lower bound and upper bound of $\alpha_2$ respectively , i.e. $L \leq \alpha_2 \leq H$.
![Alt text](image-33.png)
Based on graph above, we have
$$
\alpha_{2}^{new} =
\begin{cases}
\alpha_{2}^{new,unc} & \text{if } L \leq \alpha_{2}^{new,unc} \leq H \\
H & \text{if } \alpha_{2}^{new,unc} > H \\
L & \text{if } \alpha_{2}^{new,unc} < L
\end{cases}
$$

Our question lies at how to find $L$ and $H$.
We have $\alpha_1y_1 + \alpha_2y_2 = \xi$, and $0 \leq \alpha_1 , \alpha_2 \leq C$, so
* Case 1: $y_1 \neq y_2$ (one +1 and one -1), i.e. $\alpha_1 - \alpha_2 = y_1\xi$
  * $L = max(0, \alpha_2^{old} - \alpha_1^{old})$
  * $H = min(C, C + \alpha_2^{old} - \alpha_1^{old})$
  
      <img src="image-35.png" alt="Alt text" width="200" height="200">

* Case 2: $y_1 = y_2$ (both +1 or both -1), i.e. $\alpha_1 + \alpha_2 = y_1\xi$
  * $L = max(0, \alpha_2^{old} + \alpha_1^{old} - C)$
  * $H = min(C, \alpha_2^{old} + \alpha_1^{old})$

       <img src="image-34.png" alt="Alt text" width="200" height="200">

To update $\alpha_1$, we have
$$
\alpha_1^{new} = y_1\xi - y_1y_2\alpha_2^{(new)} =y_1(\alpha_1^{(old)}y_1 + \alpha_2^{(old)}y_2) - y_1y_2\alpha_2^{(new)}=  \alpha_1^{old} + y_1y_2(\alpha_2^{old} - \alpha_2^{new})
$$
> Multiply by $y_1y_2$ is a trick to avoid considering different cases.
### How to choose $\alpha_1$ and $\alpha_2$ in SMO
Let's get back to our constrains:
* $1 - y_{i}(w^T x_{i}+b)  - \xi_{i} \leq 0$ (1)
* $-\xi_{i} \leq 0$ (2)
* $0 \leq \alpha_i \leq C$ (3)
* $\alpha_i + \mu_i = C$ (4)
* $\alpha_i(1 - y_{i}(w^T x_{i}+b)  - \xi_{i}) = 0$ (5)
* $\mu_i\xi_i = 0$ (6)
---
* Case 1: $\alpha_i = 0$

  From (4) we know $\mu_i = C$, so $\xi_i = 0$ from (6). Then, from (1) $y_{i}(\omega\cdot x_{i}+b)\geq1 - \xi_i$ $\Rightarrow y_{i}g(x_{i})\geq1$
* Case 2: $0 < \alpha_i < C$

  From (4) we know $\mu_i \neq 0$ and $\alpha_i \neq 0$, so $\xi_i = 0$ from (6). Then, from (1) $y_{i}(\omega\cdot x_{i}+b)=1 - \xi_i$ $\Rightarrow y_{i}g(x_{i})=1$

* Case 3: $\alpha_i = C$
  
    From (5) we know $y_{i}(w^T x_{i}+b)=1 - \xi_i$, and $\xi_i \geq 0$ from (2). Then, $y_{i}(\omega\cdot x_{i}+b) =1 - \xi_i$ $\Rightarrow y_{i}g(x_{i})\leq1$

In summary, 
$$
\alpha_i =
\begin{cases}
y_ig_i(x) \geq 1 & \text{if } \alpha_i = 0 \\
y_ig_i(x) = 1 & \text{if } 0 < \alpha_i < C ~~~ \text{(lie on support vector)}\\
y_ig_i(x) \leq 1 & \text{if } \alpha_i = C
\end{cases}
$$
where $g_i(x_i) = w^T x_i + b$.

So for $\alpha_1$, we want to choose the one that violates KKT condition the most. First we scan through data point where $0 < \alpha_i < C$. If we can't find any, then we will look at all data points.

For $\alpha_2$, we want to choose the one that can make $|E_1 - E_2|$ the largest. 

---
### How to update $b$
* When $0 < \alpha_1^{new} < C$, we have


$$y_{i}g(x_{i})=1$$
i.e.
$$\sum_{i=1}^{N}{\alpha_{i}y_{i}K_{i1}}+b=y_{1}
$$
Since 
$$E_{1}=g(x_{1})-y_{1}=\sum_{i=3}^{N}{\alpha_{i}y_{i}K_{1i}}+\alpha_{1}^{old}y_{1}K_{11}+\alpha_{2}^{old}y_{2}K_{21}+b^{old}-y_{1}$$
By plugging in, we have
$$b_1^{new} = -E_1 - y_1K_{11}(\alpha_1^{new} - \alpha_1^{old}) - y_2K_{21}(\alpha_2^{new} - \alpha_2^{old}) + b^{old}$$
Similarly, when $0 < \alpha_2^{new} < C$, we have
$$b_2^{new} = -E_2 - y_1K_{12}(\alpha_1^{new} - \alpha_1^{old}) - y_2K_{22}(\alpha_2^{new} - \alpha_2^{old}) + b^{old}$$

In summary, 
$$
b^{new} = 
\begin{cases}
b_1^{new} = b_2^{new} & \text{if } 0 < \alpha_1^{new}, \alpha_2^{new} < C \\
\\

\frac{b_1^{new} + b_2^{new}}{2} & \text{if otherwise}\\
\end{cases}
$$

## Kernel Function
After introducing kernel trick, our optimization problem becomes:
$$
\operatorname*{min}_{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i \alpha_j y_i y_j K(x_i, x_j)  -\sum_{i=1}^{N} \alpha_i
$$
subject to
$$\sum_{i=1}^{N} \alpha_i y_i = 0 \\0\leqslant\alpha_{i}\leqslant C$$
where $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$

### Positive Definite Kernel Function
Suppose $K$ is a symmetric function
$$K(x^{(i)},x^{(j)}):\mathbb{R}^{n}\times\mathbb{R}^{n}\to\mathbb{R}$$
is a valid Kernel if and only if the matrix 𝐺
, called the Kernel matrix, or Gram matrix is symmetric, positive semi-definite, where Gram matrix is defined as
$$G_{ij}=K(x^{(i)},x^{(j)}) ~~~ G_{ij} ~is~(i, j)~entry $$

## Common Kernel Functions
* Linear Kernel
$$K(x_i, x_j) = x_i^T x_j$$
* Polynomial Kernel
$$K(x_i, x_j) = (x_i^T x_j + c)^d$$
* Gaussian Kernel
$$K(x_i, x_j) = \exp(-\frac{\|x_i - x_j\|^2}{2\sigma^2})$$
* String Kernel (on discrete set)
$$[\phi_{n}(s)]_{u}=\sum_{i:s(i)=u}\lambda^{l(i)}$$
where $s$ is a string, $s(i)$ is the $i$th character of $s$, $u$ is a character, $l(i)$ is the length of the longest suffix of $s$ starting at $i$ that is also a prefix of $s$, and $\lambda$ is a decay factor ($0 < \lambda \leq 1$). The kernel is defined as
$$K(s,t)=\sum_{u}\sum_{v}[\phi_{n}(s)]_{u}[\phi_{n}(t)]_{v}$$

### String Kernel Example

|         |  bi       |  bg       |  ig       |  pi       |  pg       |  ba       |  ag       |
|---------|----------|----------|----------|----------|----------|----------|----------|
| "big"   | $\lambda^2$ | $\lambda^3$ | $\lambda^2$ | 0 | 0 | 0 | 0 |
| "pig"   | 0 | 0 | $\lambda^2$ | $\lambda^2$ | $\lambda^3$ | 0 | 0 |
| "bag"   | 0 | 0 | 0 | 0 | 0 | $\lambda^2$ | $\lambda^2$ |

For example, if $\lambda^3$ in "pig" is calulated as 
$\lambda^3 = \lambda^{\text{index of 'g' - index of 'p' + 1}} = \lambda^3$.
So "pig" is mapped into a vector $[0, 0, \lambda^2, \lambda^2, \lambda^3, 0, 0]$ in 7 dimensional space.

We can use this kernel to measure the similarity between two strings. For example, the similarity between "big" and "pig" is 
$$Similarity = \frac{K("big", "pig")}{||K("pig", "pig")||\cdot||K("big", "big")||}$$
This is similar to cosine similarity.



## Reference
Reference:
[Strong Duality](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/Slides5A.pdf)  
[SMO-Stanford](https://web.stanford.edu/group/SOL/reports/Ma-SMOvsPDCOforSVM.pdf)  
[SMO-Zhihu](https://zhuanlan.zhihu.com/p/433150785)  
Handwritten Proof can be found [here](https://github.com/WangCheng0116/ML/blob/main/Concepts/SVM%20Math%20Proof.pdf)  

## Sklearn Usage
Following is sample usage from sklearn:
```python
class sklearn.svm.SVC(
            C=1.0, # C is the penalty parameter
            kernel='rbf', # kernel function
            degree=3, # degree of polynomial kernel function
            gamma='auto', # kernel coefficient
            coef0=0.0, # constant term in kernel function
            shrinking=True, # whether to use shrinking heuristic
            probability=False, # whether to enable probability estimates
            tol=0.001, # tolerance for stopping criterion
            cache_size=200, # kernel cache size
            class_weight=None, # set the parameter C of class i to class_weight[i]*C
            verbose=False, # enable verbose output
            max_iter=-1, # limit the number of iterations within solver
            decision_function_shape='ovr', # whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2)
            random_state=None)
```  
A pretty simple usage of SVM can be found [here](https://github.com/WangCheng0116/ML/blob/main/Code%20Implementation/SVM%20Binary%20Classification/SVM_Binary_Classification.ipynb)
![Alt text](image-11.png)

# Ensemble Learning
It has two aspects:
1. Bagging: train models parallelly 
2. Boosting: train models sequentially
## Bagging
Bagging stands for **Bootstrap Aggregation**. It is a technique used to reduce the variance of our predictions by combining the result of multiple classifiers modeled on different sub-samples of the same data set.
### Random Forest
![Wiki](image-8.png)  
(extracted from wikipedia)  

A very great [resource](https://carbonati.github.io/posts/random-forests-from-scratch/) that I found online. I followed through the instruction and did [this](https://github.com/WangCheng0116/ML/blob/main/Code%20Implementation/Random%20Forest/random-forest-from-stratch-adapted.ipynb)

## Boosting

### AdaBoost (Adaptive Boosting)
#### Algorithm
**Input:** Training dataset T = {($x_1$, $y_1$), ($x_2$, $y_2$), ..., ($x_N$, $y_N$)}, where $x_i$ ∈ X ⊆ $R^n$, and $y_i$ ∈ Y = {-1, +1}. 

1. Initialize the weight distribution for the training data.  
  
$$D_1 = (w_{11}, w_{12}, \ldots, w_{1i}, \ldots, w_{1N}),$$
where $w_{1i} = \frac{1}{N}$ for $i = 1, 2, \ldots, N.$. 

1. For $m = 1, 2,..., M$
	- Use $D_m$ to train and we get a classifier $G_{m}(x)$
	- The error rate of  classifier $G_{m}(x)$ is 
  $$e_m = \sum_{i=1}^N P(G_m(x_i) \neq y_i) = \sum_{i=1}^N w_{mi} I(G_m(x_i) \neq y_i)$$

	- Compute the coefficient of $G_m(x)$: 
  $$\alpha_m = \frac{1}{2} log\frac{1- e_m}{e_m}$$
	- Update weights as:  
$$w_{m+1} = (w_{m+1,1}, \ldots, w_{m+1,i}, \ldots, w_{m+1,N})$$
$$w_{m+1,i} = \frac{w_{mi}}{Z_m} \exp(-\alpha_m y_i G_m(x_i)), \quad i = 1, 2, \ldots, N$$

Here, $Z_m$ is the normalization factor.
$$Z_m = \sum_{i=1}^N w_{mi} \exp(-\alpha_m y_i G_m(x_i))$$

1. Build our final claasifier:
$$G(x) = sign(\sum_{m=1}^{M}\alpha_mG_m(x))$$
The main idea is if a certain classifer has a bad performance, which means 
$e_m$ is less 0.5, then coefficient $a_m$ will be greater than 1. Misclassified data will play a more important role in the next round of iteration

#### Example 
| Index |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  | 10 |
|-------|---- |---- |---- |---- |---- |---- |---- |---- |---- |----|
|   X   |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  9  |
|   Y   |  1  |  1  |  1  | -1 | -1 | -1 |  1  |  1  |  1  | -1 |
1. Initialization: 
$$D_1 = (w_{11}, w_{12}, ..., w_{110})$$
$$w_{1i} = 0.1 \space for  \space i = 1,2,...,10 $$
1. When $m$ = 1
- When $v = 2.5$ it gets its lowest misclassification error, so we build:
$$ G_1(x) = \begin{cases} 1 & \text{if } x < 2.5 \\ -1 & \text{if } x > 2.5 \end{cases} $$
- Misclassification Error: $$e_1 = 0.3$$
- Coefficient for $G_1(x)$: 
$$\alpha_1 = \frac{1}{2}log\frac{1 - e_1}{e_1} = 0.4236$$
- Update weights as follows:
$$D_2 = (w_{21}, w_{22}, ..., w_{210})$$
$$w_{2i} = \frac{w_{1i}}{Z_1}exp(-\alpha_1y_iG_1(x_i))$$
$$D_2 = (0.07143, ..., 0.07143, 0.16667,0.16667,0.16667, 0.07143)$$
- Current ultimate classifier is: 
$$f_1(x) = 0.4236G_1(x)$$
1. When $m = 2$:
- - When $v = 8.5$ it gets its lowest misclassification error, so we build:
$$ G_2(x) = \begin{cases} 1 & \text{if } x < 8.5 \\ -1 & \text{if } x > 8.5 \end{cases} $$
- Misclassification Error: $e_2 = 0.07143 * 3$ since index 4, 5, 6 are mislabeled and all three have the weight of 0.07143
- Coefficient for $G_1(x)$: $\alpha_2 = \frac{1}{2}log\frac{1 - e_1}{e_1} = 0.6496$
- Update weights as follows:$D_2 = (w_{31}, w_{32}, ..., w_{310})$
$$w_{2i} = \frac{w_{1i}}{Z_1}exp(-\alpha_1y_iG_1(x_i))$$
$$D_2 = (omitted)$$
- Current ultimate classifier is: 
$$f_2(x) = 0.4236G_1(x) + 0.6496G_2(x)$$
1. When $m =3$:
Same procedure applies we can have 
$$f_3(x) = 0.4236G_1(x) + 0.6496G_2(x) + 0.7514G_3(x)$$

After all this, we derive our model, which is: 
$$G(x) = sign[f_3(x)] = sign[0.4236G_1(x) + 0.6496G_2(x) + 0.7514G_3(x)]$$

#### Why $\alpha$ is like that?
We have a recursive step here: 
$$f_m(x) = f_{m-1}(x) + \alpha_mG_m(x)$$  
By defining exponential loss function:
$$L(y, f(x)) = exp(-yf(x))$$ 
We want to find $\alpha_m$ which can minimize the overall cost function, i.e.:
$$\alpha_m = \argmin_{\alpha_m} \sum_{i=1}^N e^{-y_i(f_{m-1}(x_i) + \alpha_m G_m(x_i))}$$
it is equivalent to: 
$$\alpha_m = \argmin_{\alpha_m} \sum_{i=1}^N e^{-y_if_{m-1}(x_i) }\cdot e^{-y_i\alpha_m G_m(x_i)}$$  
Let's denote $w_{mi} = e^{-y_if_{m-1}(x_i)}$, then we have:
$$\alpha_m = \argmin_{\alpha_m} \sum_{i=1}^N w_{mi}\cdot e^{-y_i\alpha_m G_m(x_i)}$$  
which is 
$$\argmin_{\alpha_m} \sum_{y_i = G_m(x_i)}w_{mi}\cdot e^{-\alpha_m} + \sum_{y_i \neq G_m(x_i)}w_{mi}\cdot e^{\alpha_m}$$  
since $y_i$ and $G_m(x_i)$ are either 1 or -1. By furtuer simplification, we have:  
$$\argmin_{\alpha_m}e^{-\alpha_m}(\sum_{i =1}^{N}w_{mi} - \sum_{y_i \neq G_m(x_i)}w_{mi}) + \sum_{y_i \neq G_m(x_i)}w_{mi}\cdot e^{\alpha_m} $$

$$= \argmin_{\alpha_m}e^{-\alpha_m}\sum_{i =1}^{N}w_{mi} + (e^{\alpha_m}- e^{-\alpha_m})\sum_{y_i \neq G_m(x_i)}w_{mi}$$
Since we want to find $\alpha_m$ which can minimize the overall cost function, we can take the derivative of the above equation w.r.t $\alpha_m$ and set it to 0. Then we have:
$$-e^{-\alpha_m}\sum_{i =1}^{N}w_{mi} + (e^{\alpha_m}+ e^{-\alpha_m})\sum_{y_i \neq G_m(x_i)}w_{mi} = 0$$  
$$\Leftrightarrow e^{2\alpha _m} = \frac{\sum_{y_i = G_m(x_i)}w_{mi}}{\sum_{y_i \neq G_m(x_i)}w_{mi}}$$  
$$\Leftrightarrow \alpha _m = \frac{1}{2}log\frac{1- e_m}{e_m}$$
#### Why we can update $w_m$ is like that?
$$w_{m+1i} = e^{-y_if_{m}(x_i)}$$  
$$=e^{-y_i(f_{m-1}(x_i)+ \alpha_mG_m(x_i))}$$  
$$= e^{-y_if_{m-1}(x_i)}\cdot e^{-y_i\alpha_mG_m(x_i)}$$  
$$= w_{mi}\cdot e^{-y_i\alpha_mG_m(x_i)}$$  
but here we need to introduce a normalization factor $Z_m$ to make sure $w_{m+1}$ is a distribution, i.e., in next round of iteration, the sum of it will be 1.
### Boosting Decision Tree
#### Herustic Example
Suppose we have a train data set of
| $x_i$  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
|-----|------|------|------|------|------|------|------|------|------|------|
| $y_i$  | 5.56 | 5.70 | 5.91 | 6.40 | 6.80 | 7.05 | 8.90 | 8.70 | 9.00 | 9.05 |
1. At the first iteration:
Suppose we let $s = 6.5$ be the threshold, then we have:
$$T_1(x) = \begin{cases} 6.24 & \text{if } x < 6.5 \\ 8.91 & \text{if } x > 6.5 \end{cases} $$
Then we can calculate the MSE:
$$MSE_1 = \frac{1}{10}\sum_{i=1}^{10}(y_i - T_1(x_i))^2 = 1.93$$
and it happens to be the least MSE we can get among all possible thresholds (9 in total in this case).  
Then we derive the residual:
$$r_{2i} = y_i - T_1(x_i)$$  
which can be shown in this table:
| $x_i$  | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
|-----|------|------|------|------|------|------|------|------|------|------|
| $r_{2i}$  | -0.68 | -0.54 | -0.33 | 0.16 | 0.56 | 0.81 | -0.01 | -0.21 | 0.09 | 0.14 |
2. At the second iteration:
We treat $r_{2i}$ as our new $y_i$ and we want to find a new $T_2(x)$ to minimize the MSE. We can find:
$$T_2(x) = \begin{cases} -0.52 & \text{if } x < 3.5 \\ 0.22 & \text{if } x > 3.5 \end{cases} $$  
By combining this with previous learner, we have:
$$f_2(x) = T_1(x) + T_2(x) = \begin{cases} 5.72 & \text{if } x < 3.5 \\ 7.46 & \text{if } 3.5 < x < 6.5 \\ 9.13 & \text{if } x > 6.5 \end{cases} $$

At last, after MSE has met the given tolerance, we sum all the trees up and get our final model:
$$f_n(x) = \sum_{m=1}^{M}T_m(x)$$
> A quick note here: We are trying to approximate the residual at each iteration, which is the key point of boosting trees.

### Gradient Boosting Decision Tree 
#### Why Gradient?
Key idea: Gradient is an approximation of the residual.  

Math Proof:  
We know $f(x)$'s first order derivative at $x = x_{t-1}$ is：
$$f(x) \approx f(x_{t-1}) + f^{'}(x_{t-1})(x-x_{t-1})$$  
Similarly, Loss Function $L(y,F(x))$'s taylor expansion at $F(x) = F_{t-1}(x)$ is ：
$$L(y,F(x))\approx L(y, F_{t-1}(x)) + \left[ \frac{\delta L(y,F(x))}{\delta F(x)} \right]_{F(x) = F_{t-1}(x)}(F(x)-F_{t-1}(x))$$ 
By subsituting $F(x) = F_{t}(x)$, we have：
$$L(y,F_{t}(x))\approx L(y, F_{t-1}(x)) + \left[ \frac{\delta L(y,F(x))}{\delta F(x)} \right]_{F(x) = F_{t-1}(x)}(F_{t}(x)-F_{t-1}(x))$$
So $-\left[ \frac{\delta L(y, F(x))}{\delta F(x)} \right]_{F(x)=F_{t-1}(x)}$ is an approximation of $y-f_{t-1}(x)$ = $residual$
#### Algorithm
![Alt text](image-21.png)
(credits to [wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting#Algorithm))
### XGBoost
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. It stands for eXtreme Gradient Boosting.

#### Loss Function Formulation
We know that XGBoost is a sum of $k$ basic models:
$$\hat{y}_i = \sum_{t=1}^kf_{t}(x_{i})$$  
where $f_{k}$ is the $k_{th}$ model and $\hat{y}_{i}$ is the prediction of $i_{th}$ sample.
The loss function is a summation over all datas:
$$L =\sum_{i=1}^nl(\hat{y}_i , y_{i})$$  
$n$ is the size of data.
By introducing regularization, we have
$$L =\sum_{i=1}^nl(\hat{y}_i , y_{i}) + \sum_{i=1}^k\Omega(f_{i})$$  

Since XGBoost uses a strategy where each weak learner is built upon the previous one, i.e.,
$$\hat{y}_i ^{t} = \hat{y}_i ^{t - 1} + f_{t}(x_{i})$$

So the Loss Function at step $t$ can be updated as:
$$L^{(t)} =\sum_{i=1}^nl(\hat{y}_i ^{t - 1} + f_{t}(x_{i}), y_{i}) + \sum_{i=1}^t\Omega(f_{i})$$  

By Taylor's Theorem:
$$f(x + \Delta x) \approx f(x ) + f^{'}(x) \Delta x+ \frac{1}{2}f^{''}(x)\Delta x^{2}$$
By treating $\hat{y}_i ^{t - 1}$ as $x$, $f_{t}(x_{i})$ as $\Delta x$, we have:
$$
L^{(t)} = \sum_{i=1}^{n} [l(y_i, \hat{y}^{t-1}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \sum_{i=1}^{t} \Omega(f_i)
$$
Note that $g_{i}$ and $h_{i}$ are first and second derivative w.r.t to **$\hat{y}_i ^{t - 1}$** respectively.  
Take MSE as an example:
$$
\sum_{i=1}^{n} (y_i - (y_i^{t-1} + f_t(x_i))^2)
$$
Then, 
$$
g_i = \frac{\partial (\hat{y}^{t-1} - y_i)^2}{\partial y^{t-1}} = 2(\hat{y}^{t-1} - y_i)
$$

$$
h_i = \frac{\partial^2(\hat{y}^{t-1})}{\partial\hat{y}^{t-1}} = 2
$$

Since at step $t$  we already know the value of $\hat{y}^{t-1}$, so $l(y_i, \hat{y}^{t-1})$ is a contant, which doesn't have an impact on our loss function.  

$$
L^{(t)} \approx \sum_{i=1}^{n} [g_if_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \sum_{i=1}^{t} \Omega(f_i)
$$

we can define $f_t(x) = w_{q(x)}$, where $x$ is a sample data, $q(x)$ indicates which leaf this data falls into, and $w_{q(x)}$ represents the value of that leaf.  

The complexity of the decision tree can be decided from two aspects, 1. total number of leaves $T$; 2. weight of leaves. And both of them shouldn't be too large.  Hence, we derive the regularization term as:
$$
\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{i=1}^{T} w_j^2
$$
![Alt text](image-12.png)
Let  $I_j = \{ i \,|\, q(x_i) = j \}$ be the $j_{th}$ be a set of indices where each index of sample data falls in this leave. (叶子节点所有instances的下标), then:
$$ L^{(t)} \approx \sum_{i=1}^{n} \left[g_if_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)\right] + \Omega(f_t) \\ = \sum_{i=1}^{n} \left[g_i w_{q(x_i)} + \frac{1}{2} h_i w^2_{(x_i)}\right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 \\ = 
\sum_{j=1}^{T}\left[ (\sum_{i\in I_j}^{}g_i)w_j + \frac{1}{2} (\sum_{i\in I_j}^{} h_i + \lambda) w_j^2 \right]+ \gamma T $$
since $f_t(x_i) = w_{q(x_i)}$, what the last equation essentially does is  convert $x_1+x_2+x_3+x_4+x_5+...$ to $(x_1+x_2)+(x_3+x_4+x_5)+...$, representing the iteration happening on leaves.  

In order to simplify, let $G_j = \sum_{i \in I_j} g_i$ and $H_j = \sum_{i \in I_j}^{} h_i$ , then:

$$L(t) = \sum_{j=1}^T [G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 ] + \gamma T $$

To minimize $G_jw_j+\frac{1}{2}( H_j+\lambda) \cdot w_j^2$ and find  $w_j$, it is equivalent of 
$$w_j = argmin \space G_j w_j+\frac{1}{2}( H_j+\lambda) \cdot w_j^2$$
Since it's a quadratic funtion, we can derive 
$$w_j = -\frac{G_j}{ H_j+\lambda}$$
By pluggin back, we have,
$$L(t) =  -\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{ H_j+\lambda}+\gamma T$$ 
![Alt text](image-14.png) 
It is easier when looking at real data. We can think of $G_j$ as the sum of the first derivative of the loss function of the data points that fall into the $j_{th}$ leaf. And $H_j$ is the sum of the second derivative of the loss function of all the data points that fall into the $j_{th}$ leaf.

#### Partitioning (How to find the best split)
Let's first take a step back and think about what we have achieved so far. We only get one thing by far, which is the loss function:
$$L(t) =  -\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{ H_j+\lambda}+\gamma T $$

But how can we build a tree which can minimize this loss function? One feasible way is to try all possible splits and find the best one, i.e. exhaustion. But this is not efficient. So we need to find a way to find the best split in a more efficient way.

##### Greedy Algorithm
Before splitting, the loss function is:
$$L_{before} = -\frac{1}{2} \left[ \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] + \gamma$$

After splitting:
$$L_{after} = -\frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} \right] + 2\gamma$$

The gain can be expressed as:
$$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right]- \gamma $$
![Alt text](image-15.png)
Above is an intuitive elaboration of the gain. (It is not exactly the same but the idea is similar)
This calls out the algorithm:
![Alt text](image-16.png)  
In English, in each iteration, choose the split that can let loss function decrease the most.

##### Approximate Algorithm
![Alt text](image-17.png)
- First for loop：Find a partition of sorted data set based on different percentile.
- Second for loop: Compute the loss for each bucket and find the best split after comparison.
This is an example of 1/3 percentile split.
![Alt text](image-18.png)


# K Nearest Neighbors (KNN) Algorithm
At its core, KNN classifies or predicts the target value of a new data point by examining the k-nearest data points from the training dataset. The $k$ in KNN represents the number of nearest neighbors considered for making predictions. When classifying a data point, KNN counts the number of neighbors belonging to each class and assigns the class label that is most common among the k-nearest neighbors.
> k can be chosen by cross validation. It is usually an odd number to avoid ties and less than the square root of the amount of data points.



## Algorithm I - Linear Scan
```
For every point x in the dataset: -- O(N)
    Compute distance between x and query point q -- O(d)
    Return the k points that are nearest to q -- O(k) (By quick select)
```
> Time Complexity: $O(Nd + Nk)$, where $N$ is number of data points, $d$ is dimension of data, $k$ is number of neighbors.
## Algorithm II - KD Tree
### Build KD Tree
1. Constructing a root node:  
  We first choose we will spilt the whole dataset by the first coordinate. Then, we find the median of the first coordinate and use it as the root node and partition the dataset into two parts.  
2. Repeat:  
Suppose we are at depth $j$, we now consider the $j$(mod $k$) + coordinate and do the same thing. This terminates when we have no more data points to split.
> Time Complexity: $O(dN\log N)$, where $N$ is number of data points, $d$ is dimension of data.  

<img src="image-32.png" width="300" height="400">

### Search KD Tree to find k nearest neighbors
```
Find nearest neighbor:
For Each Point:
- Start at the root
- Traverse the Tree to the section where the new point belongs
- Find the leaf; store it as the best
- Traverse upward, and for each node:
  - If it’s closer, it becomes the best
  - Check if there could be yet better points on the other side:
    - Fit a sphere around the point of the same radius as the distance to the current best
    - See if that sphere goes over the splitting plane associated with the considered branch point
    - If there could be, go down again on the other side. Otherwise, go up another level
```
> Time Complexity: $O(d\log N)$, where $d$ is dimension of data, $N$ is number of data points.
```
Find k nearest neighbors:
For i = 1 to k:
  Find nearest neighbor
  Remove it from the tree
```
> Time Complexity: $O(kd\log N)$, where $d$ is dimension of data, $N$ is number of data points, $k$ is number of neighbors.

[Visualization of Searching a KD Tree | 8:08](https://www.bilibili.com/video/BV1No4y1o7ac?p=22&vd_source=43fae35e0d515715cd36645ea2e6e547)
