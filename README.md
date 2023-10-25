# Machine Learning Daily Blog
> **⚠️For a better viewing experience, I recommend reading markdown file in vscode editor instead of on GitHub. GFM doesn't render content exactly the same with vscode.**
# Machine Learning Topics I have Learned
Notes: [Supervised Learning](https://github.com/WangCheng0116/ML/blob/main/Supervised%20Learning/Supervised%20Learning.md) | [Unsupervised Learning](https://github.com/WangCheng0116/ML/blob/main/Unsupervised%20Learning/Unsupervised%20Learning.md) | [Reinforment Learning](https://github.com/WangCheng0116/ML/blob/main/Reinforcement%20Learning/Reinforcement_Learning_Note.md)
## Decision Tree

- **ID3 (Iterative Dichotomiser)**
- **C4.5**
- **CART**
  - *Gini Index*
  - *Dealing with continuous values*
- **Pre-Pruning**
- **Post-Pruning**
  - *Reduced Error Pruning (REP)*
  - *Pessimistic Error Pruning (PEP)*
  - *Minimum Error Pruning (MEP)*
  - *Cost-Complexity Pruning (CCP)*

## Bayes Classifier

- **Bayes Theorem**
- **Naive Bayes Theorem**
- **Bayes Classifier**
- **Native Bayes Classifier**

## Linear Regression

### Linear Regression with One Variable

- **Gradient Descent**
- **Applying Gradient Descent to Linear Regression Model**

### Linear Regression with Multiple Variables

- **New Notations**
- **Gradient Descent for Multiple Variables**
- **Python Implementation**
- **Feature Scaling**
- **Learning Rate**

## Features and Polynomial Regression

## Normal Equation

## Comparison

- **What if X^T X is not invertible?**

## Logistic Regression

- **Classification Problems**
- **Decision Boundary**
- **Cost Function**
- **Vectorization of Logistic Regression**
- **Multiclass Classification**

## Regularization

- **Underfit and Overfit**
- **Motivation for Regularization**
- **Gradient Descent and Normal Equation Using Regularization in Linear Regression**
- **Gradient Descent Using Regularization in Logistic Regression**
- **Gradient Descent Using Regularization in Neural Networks**
- **Dropout (In Neural Networks)**
  - *Implementation*
- **Other Methods of Regularization**

## Neural Networks

- **Activation Functions**
- **Multiclass Classification**
- **Cost Function**
- **Backpropagation Algorithm**
- **Random Initialization**

## Deep Neural Network

- **Working Flow Chart**

## Setting Up ML Application

- **Data Split: Training / Validation / Test Sets**
- **Vanishing and Exploding Gradients**
- **Gradient Checking (Grad Check)**

## Optimization Algorithms

- **Mini-Batch Gradient Descent**
- **Exponentially Weighted Averages**
- **Bias Correction in Exponentially Weighted Averages**
- **Gradient Descent with Momentum**
- **RMSprop (Root Mean Square Propagation)**
- **Adam Optimization Algorithm (Adaptive Moment Estimation)**

## Hyperparameter Tuning, Batch Normalization, and Programming Frameworks

- **Tuning Process**
- **Using an Appropriate Scale to Pick Hyperparameters**
- **Hyperparameters Tuning in Practice: Pandas vs. Caviar**
- **Normalizing Activations in a Network**
- **Batch Normalization (BN)**
- **Fitting Batch Norm into a Neural Network**
- **Batch Normalization and Its Usage**
- **Why Does Batch Norm Work?**
- **Benefits of Batch Normalization**
- **Batch Norm at Test Time**

## ML Strategy

- **Orthogonalization in Machine Learning**
- **Training / Validation / Test Set Split**
- **Distribution of Validation and Test Sets**
- **Partitioning Data**
- **Comparing to Human Performance**
- **Summary**

## Convolutional Neural Networks

- **Padding in Convolution**
- **Strided Convolution**
- **Convolution in High Dimensions**
- **Single-Layer Convolutional Network**
  - *Summary of Notation*
- **Simple Convolutional Network Example**
- **Pooling Layer**
- **Fully Connected Layer**
- **Example of a Convolutional Neural Network (CNN)**
- **Why Convolutions?**

## Convolutional Neural Networks (CNN) Architectures

- **LeNet-5**
- **AlexNet**
- **VGG**
- **ResNet (Residual Network)**
  - *Reasons Why Residual Networks Work**

## Object Detection

- **Object Localization**
- **Object Detection**
- **Convolutional Implementation of Sliding Windows (Do partition all at once, no need to feed in sliding windows sequentially)**
- **Bounding Box Predictions**
- **Intersection Over Union (IoU)**
- **Non-Maximum Suppression**
- **R-CNN**

## Face Recognition and Neural Style Transfer

- **Difference Between Face Verification and Face Recognition**
- **One-Shot Learning**
- **Siamese Network**
- **Triplet Loss**
- **As Binary Classification**
- **Neural Style Transfer**
- **What Deep Convolutional Networks Learn**
- **Cost Function For Neural Style Transfer**
  - *Content Cost Function*
  - *Style Cost Function*

## Sequence Models

- **Notations**
- **Recurrent Neural Network (RNN) Model**
  - *Forward Propagation*
  - *Backpropagation Through Time (BPTT)*
- **Different Types of RNNs**
- **Language Model**
- **Sampling**
- **The Gradient Vanishing Problem in RNNs**

## Expectation Maximization (EM) Algorithm

- **Intuitive Explanation**
- **Intuitive Algorithm**
- **Formal Proof**
  - *Likelihood Function*
  - *Log Likelihood Function*
  - *Jensen Inequality*
  - *EM*
- **EM Algorithm**
- **Why EM Converges?**

## Gaussian Mixture Model (GMM)

- **Notation**
- **Algorithm**
- **Intuitive Example**
- **Implementation**
- **Math Proof**

## Clustering

- **Measuring Similarity**
  - *Minkowski Distance:*
  - *Mahalanobis Distance:*
  - *Correlation Coefficient:*
  - *Cosine Similarity:*

## K-Means Clustering

- **Algorithm**
- **How to Choose k?**
- **Improvement**
- **Smarter Initialization - K-Means++**
- **Why K-Means is a Variant of EM?**

## K-Nearest Neighbors (KNN) Algorithm

- **Algorithm I - Linear Scan**
- **Algorithm II - KD Tree**
  - *Build KD Tree*
  - *Search KD Tree to Find k Nearest Neighbors*

## Support Vector Machine

- **Proof**
  - *Problem Formulation*
  - *Optimization Problem*
- **Duality Conversion**
- **Algorithm (Hard Margin)**
- **Soft Margin**
- **Algorithm (Soft Margin)**
- **Another Perspective: Hinge Loss**
- **SMO (Sequential Minimal Optimization)**
  - *How to Choose $\\alpha_1$ and $\\alpha_2$ in SMO*
  - *How to Update $b$*
- **Kernel Function**
  - *Positive Definite Kernel Function*
- **Common Kernel Functions**
  - *String Kernel Example*
- **Reference**
- **Sklearn Usage**

## Ensemble Learning

- **Bagging**
  - *Random Forest*
- **Boosting**
  - *AdaBoost (Adaptive Boosting)*
    - *Algorithm*
    - *Example*
    - *Why $\\alpha$ is like that?*
    - *Why we can update $w_m$ is like that?*
  - *Boosting Decision Tree*
    - *Heuristic Example*
  - *Gradient Boosting Decision Tree*
    - *Why Gradient?*
    - *Algorithm*
  - *XGBoost*
    - *Loss Function Formulation*
    - *Partitioning (How to find the best split)*
      - *Greedy Algorithm*
      - *Approximate Algorithm*

## Bellman Equation

- **Motivating Example**
- **State Value**
- **Bellman Equation**
- **Bellman Equation in Matrix Form**
- **Solving Bellman Equation (Closed Form and Iterative Method)**
- **Action Value**
- **Summary**

## Optimal State Value and Bellman Optimality Equation (BOE)

- **Optimal Policy**
- **Bellman Optimality Equation (BOE)**
- **Contraction Mapping Theorem**
  - *Definition*
  - *Theorem*
  - *Proof*
- **BOE is a Contraction Mapping**
- **Solution of BOE**
- **$\\gamma$ in BOE**
- **Value Iteration Algorithm**
  - *Example*
- **Policy Iteration Algorithm**
  - *Example*
- **Truncated Policy Iteration**
  - *Algorithm*
- **Summary**

## Monte Carlo Method

- **Mean Estimation and Law of Large Numbers**
- **MC Basic**
  - *Algorithm:*
  - *Example*
- **MC Exploring Starts**
  - *Making Sample More Efficient*
  - *Making Update More Efficient*
  - *Algorithm*
- **MC $\\epsilon$-Greedy**
  - *Soft Policy*
  - *Algorithm*
  - *How Does $\\epsilon$ Affect?*
- **Summary**


## Challenges
  - [Matrix Differentiation](https://github.com/WangCheng0116/ML/blob/main/Challenges/Matrix%20Differentiation.md)
  - [Backpropagation](https://github.com/WangCheng0116/ML/blob/main/Challenges/Back%20Propogation.md)
  - [SVM Math Proof](https://github.com/WangCheng0116/ML/blob/main/Challenges/SVM%20Math%20Proof.md)
  - [XGBoost Math Proof](https://github.com/WangCheng0116/ML/blob/main/Challenges/XGBoost.md)
  - [EM Proof](https://github.com/WangCheng0116/ML/blob/main/Challenges/EM.md)
  
## Code (code implementation for model)
  - [CNN](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/CNN)
  - [KNN-Iris](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/KNN-Iris)
  - [Optimization Method - Mini-batches, Momentum and Adam](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Optimization%20Method%20-%20Mini-batches%2C%20Momentum%20and%20Adam)
  - [CNN - Fruit Recognition](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/CNN%20-%20Fruit%20Recognition)
  - [Linear Regression](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Linear%20Regression)
  - [Random Forest](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Random%20Forest)
  - [Decision Tree](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Decision%20Tree)
  - [Logistic Regression](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Logistic%20Regression)
  - [Regularization - L2 and dropout](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Regularization%20-%20L2%20and%20dropout)
  - [Deep Neural Network](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Deep%20Neural%20Network)
  - [Neural Network - Binary Classification on 2D plane](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Neural%20Network%20-%20Binary%20Classification%20on%202D%20plane)
  - [ResNets](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/ResNets)
  - [GradientChecking](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/GradientChecking)
  - [Neural Network - Digit Recognition](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/Neural%20Network%20-%20Digit%20Recognition)
  - [Choose Good Initialization](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/choose%20good%20Initialization)
  - [SVM on Binary Classification](https://github.com/WangCheng0116/ML/blob/main/Code%20Implementation/SVM%20Binary%20Classification/SVM_Binary_Classification.ipynb) 
## Resources
  - [Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html) 
  - [Neural Network Playground](https://playground.tensorflow.org/)
  - [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
  - [LaTeX Cheatsheet](https://wch.github.io/latexsheet/latexsheet.pdf)
  - [GAN Lab](https://poloclub.github.io/ganlab/)
  
  

## Calendar 

### September 2023  
|   Sunday   |   Monday   |   Tuesday   |   Wednesday   |   Thursday   |   Friday   |   Saturday   |
|:---------:|:---------:|:---------:|:-----------:|:---------:|:-------:|:---------:|
|           | [ ] 1     | [ ] 2     | [ ] 3       | [ ] 4     | [ ] 5   | [ ] 6     |
| [ ] 7     | [ ] 8     | [ ] 9     | [ ] 10      | [ ] 11    | [ ] 12  | [ ] 13    |
| [ ] 14    | [ ] 15    | [ ] 16    | [ ] 17      | [ ] 18    | [ ] 19  | [ ] 20    |
| [ ] 21    | [ ] 22    | [ ] 23    | [ ] 24      | [ ] 25    | [ ] 26  | [ ] 27    |
| [✅ 28](##28-Sept) | [✅ 29](##29-Sept) | [✅ 30](##30-Sept) |           |         |           |

### October 2023  
|   Sunday   |   Monday   |   Tuesday   |   Wednesday   |   Thursday   |   Friday   |   Saturday   |
|:---------:|:---------:|:---------:|:-----------:|:---------:|:-------:|:---------:|
|           | ✅ [1](##1-Oct)     |  ✅ [2](##2-Oct)     |  ✅ [3](##3-Oct)       | ✅ [4](##4-Oct)     | [Midterm] 5   |  ✅[6](##6-Oct)     |
|  ✅[7](##7-Oct)      | [Midterm] 8     | [Midterm] 9     | ✅[10](##10-Oct)       | ✅[11](##11-Oct)    | ✅[12](##12-Oct)  | ✅[13](##13-Oct)    |
| ✅[14](##14-Oct)   | [Chill] 15    | ✅[16](##16-Oct)    | ✅[17](##17-Oct)      | ✅[18](##18-Oct)    | ✅[19](##19-Oct)  | ✅[20](##20-Oct)    |
| ✅[21](##21-Oct)   | ✅[22](##22-Oct)     | ✅[23](##23-Oct)     | [ ] 24      | [ ] 25    | [ ] 26  | [ ] 27    |
| [ ] 28    | [ ] 29    | [ ] 30    | [ ] 31      |           |         |           |


## 28 Sept  
Today is officially the first day of my journey in Machine Learning. During the conversation with Prof Soh, he recommended I first learn the concept of random forest and gradient boosted, but I feel like I might start from the whole picture of ML and then dive into those two topics meticulously. The material that I opted for is from Andrew Ng, a pretty well-known course. And today I managed to finish the following chapters. I took special notice of math formulas, so I wrote all the steps in a rigorous mathematical manner. 

* Linear Regression with one variable
* Matrix Review
* Linear Regression with multiple variables
* Features and Polynomial Regression
* Logistic Regression

[Notes can be found here](https://github.com/WangCheng0116/ML/blob/main/AndrewNgMLCourse/note.md)

## 29 Sept
I GOT DRIVEN CRAZY BY BP PROOF!!!  
I have watched at least five different videos and none of them explain this thoroughly, in the end, I have no choice but to use plain math. What torture it is(I hate all those subscripts and superscripts in NN. HATE THEM!).

Today what I have learned:
* Regularization
* Neural Networks
* Deep Neural Networks

[Notes can be found here](https://github.com/WangCheng0116/ML/blob/main/AndrewNgMLCourse/note.md)

## 30 Sept
Today I did something different, something real!   
This is my first neural network for recognizing digits. Details can be found [here](https://github.com/WangCheng0116/ML/tree/main/Projects/Digit%20Recognition), I also include my takeaway there.  

Other topics that I have learned:  
- Mini-Batch Gradient Descent

- Exponentially Weighted Averages

- Bias Correction in Exponentially Weighted Averages

- Gradient Descent with Momentum

- RMSprop

- Adam Optimization Algorithm

- Learning Rate Decay

- Local Optima Issues

## 1 Oct
Today is the last day of recess week. From tomorrow on I think I will shift my focus to midterms again. Anyway, this is what I have learned, and it was the first time to have been exposed to CNN, the thing only existed in legend before. But it turned out to be quite easy to understand due to its high resemblance with NN.
- Orthogonalization in Machine Learning
- Training / Validation / Test Set Split
- Distribution of Validation and Test Sets
- Partitioning Data
- Comparing to Human Performance
- Convolutional Neural Networks
- Padding in Convolution
- Strided Convolution
- Convolution in High Dimensions
- Single-Layer Convolutional Network
- Simple Convolutional Network Example
- Pooling Layer
- Fully Connected Layer
- Example of a Convolutional Neural Network (CNN)
- Why Convolutions?


## 2 Oct
Conception wise:
- Padding in Convolution
- Strided Convolution
- Convolution in High Dimensions
- Single-Layer Convolutional Network
- Pooling Layer
- Fully Connected Layer (FC)
- Example of a CNN
- Why Convolutions?
- CNN Architectures
- Reasons Why Residual Networks Work

Projects wise:
[NN Binary Classification](https://github.com/WangCheng0116/ML/tree/main/Projects/Neural%20Network%20-%20Binary%20Classification%20on%202D%20plane)


## 3 Oct
The online tutorial became more and more theoretical, so I decided to include more code implementation in my learning process.  
Still, conceptwise:
- Object Detection
  - Bounding Boxes
  - Target Labels
  - Loss Functions
  - Sliding Windows Detection Algorithm
  - Convolutional Implementation of Sliding Windows
  - YOLO Algorithm
  - Intersection Over Union (IoU)
  - Non-Maximum Suppression (NMS)
  - R-CNN
  - Anchor Boxes
  
Projects wise:  
[Logistic Regression](https://github.com/WangCheng0116/ML/tree/main/Projects/Logistic%20Regression)
[Multiple Methods for initialization](https://github.com/WangCheng0116/ML/tree/main/Projects/choose%20good%20Initialization)  

## 4 Oct
Conceptwise:
- Difference Between Face Verification and Face Recognition
- One-Shot Learning
- Siamese Network
- Triplet Loss
- As Binary Classification
- Neural Style Transfer
- What Deep Convolutional Networks Learn
- Cost Function For Neural Style Transfer
- Content Cost Function
- Style Cost Function

Projects wise:
[DNN](https://github.com/WangCheng0116/ML/tree/main/Projects/Deep%20Neural%20Network)
[Gradient Checking](https://github.com/WangCheng0116/ML/tree/main/Projects/GradientChecking)

## 5 Oct
Midterm Time  

## 6 Oct
Project:  
[Regularization-L2 and dropout](https://github.com/WangCheng0116/ML/tree/main/Projects/Regularization%20-%20L2%20and%20dropout)


## 7 Oct
Project:
[Optimization Method: Mini-batches, Momentum and Adam](https://github.com/WangCheng0116/ML/tree/main/Projects/Optimization%20Method%20-%20Mini-batches%2C%20Momentum%20and%20Adam)

## 10 Oct  
Finally, the midterm week has ended. It's time to get back on track.  
I did a project on Kaggle about [fruit recognition](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/CNN%20-%20Fruit%20Recognition)  
In terms of new topics, I opted for a relatively easier one -- KNN, my own implementation from scratch and application on Iris dataset can be found [here](https://github.com/WangCheng0116/ML/tree/main/Code%20Implementation/KNN-Iris)

## 11 Oct
Plan for going through new concepts has been put aside because I saw the syllabus that I will focus on NLP, which I think is not that relevant to my current job. So..., see you later NLP.  

Instead, I will focus more on using ML framework, more specifically, Pytorch. Notes can be found [here](https://github.com/WangCheng0116/ML/blob/main/pytorch/note.md)  
Projects done:  
* [CNN-Multi-classification](https://github.com/WangCheng0116/ML/blob/main/pytorch/code/PyTorch_NN_MultiClassification.ipynb)
* [Fashion MNIST](https://github.com/WangCheng0116/ML/blob/main/pytorch/code/Pytorch_FashionMnist.ipynb)

## 12 Oct
Things have started getting busier and busier recently, mainly because of CS2103T and stuff :( As a result, today I learned two relatively chill topics, which are tensorboard and transform. I didn't have time to actually implement them, but I will mark this down and try to use them in the future. Note can be found [here](https://github.com/WangCheng0116/ML/blob/main/pytorch/note.md)

## 13 Oct
Today I tried digging into a gradient-boosted method, which is stated to be quite important by Prof. Again, this topic is very mathy and for future actual code usage, I will try to look for suitable frameworks and train this model. This topic deserves to be included in the [challenge](https://github.com/WangCheng0116/ML/blob/main/Challenges/XGBoost.md) folder.
## 14 Oct
This is a hard day. Currently, I need to confess that SVM is by far the most math-heavy topic I have ever learned. On a side note, it is very interesting to see how my math courses are intertwined with SVM, because I happen to be learning non-linear optimization (MA3236). I will put SVM in [challenge](https://github.com/WangCheng0116/ML/blob/main/Challenges/SVM%20Math%20Proof.md) folder. A very simple implementation of SVM can be found [here](https://github.com/WangCheng0116/ML/blob/main/Code%20Implementation/SVM%20Binary%20Classification/SVM_Binary_Classification.ipynb) (using scikit)

## 16 Oct
Today's topic is a quite unique one, since we are going back to revise the concept of decision tree. Things turned out to be so surprising -- 2109S only covers a very tiny small portion of that and omit a lot of math induction. Learned a lot :)

## 17 Oct
Today's topic is about ensembling method, which is a very popular method in ML. And also, the method called gradient boosting has been emphasized by prof. Indeed it is a very powerful method, but it is also very mathy. Refer to [here](https://github.com/WangCheng0116/ML/blob/main/Concepts/note.md#ensemble-learning) for more details. 

## 18 Oct
Another pretty mathy day. After enterting the realm of unsupervised learning, concepts of probability and statistics start kicking in more frequently. Today's topic is about GMM, EM and a bit more advanced stuff about K-mean. EM is very smart and somehow even quite intuitive, but the math behind it is very intimidating. Refer to [here](https://github.com/WangCheng0116/ML/blob/main/Challenges/EM.md) for more details.

## 19 Oct
Too much information has been flooding in my mind. Pause for a while not to take in new concepts and take a deep breath. Today's task - Review!

## 20 Oct
Today I revisited the concept of SVM after I learned the corresponding knowledge in MA3236 (non-linear optimization), and specifically, KKT and duality. Also I have included a muc h more rigorous proof of SVM.

## 21 Oct
Today's topic is about SVD. It is very amazing to see how the knowledge of linear algebra can be applied to ML. SVD is a powerful tool for dimensionality reduction, and it is also the foundation of PCA, which I plan to study tomorrow.

## 22 Oct
PCA is a powerful tool to reduce dimensionality. And the math basis is SVD. The idea behind is understandable, but I really didn't dive into the application part. In what scenario should we use PCA? I will explore it in the future.

## 23 Oct
I skip to anohter new topic lol. Reinforcement learning. Simply because this is the topic that I am interested in. Bellman Equation is yet the most confusing part I have ever encountered in machine learning. It took me a very long time and a lot of efforts refering to different sources to understand it. 

## 24 Oct
In most cases, it is not possible to solve Bellman Equation analytically. So we need to resort to numerical methods. Monte Carlo is one of them. The main idea behind is to use sampling to estimate the value function.

## 25 Oct
Today I realized I skipped the topic of Perceptron and it happened to be the topic of recent lecture. So I went back and learned it. It is a very simple model, but it is the foundation of neural network. And I guess also a bit of SVM? It basically looks a degenarated version of SVM.

## 26 Oct
Math Proof of GMM.

## HMM