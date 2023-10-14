# Machine Learning Daily Blog
![image](https://github.com/WangCheng0116/ML/assets/111694270/df7c767e-f08d-4ed1-a189-3c623519b105)

## Overview  
- Basic Concepts ([notes](https://github.com/WangCheng0116/ML/blob/main/Concepts/note.md))
  - Linear Regression with one variable
  - Matrix Review
  - Linear Regression with multiple variables
  - Features and Polynomial Regression
  - Normal Equation
  - Comparison
  - Logistic Regression
  - Regularization
  - Neural Networks
  - Deep Neural Network
  - Setting Up ML Application
  - Optimization Algorithms
  - Hyperparameter Tuning Batch Normalization and Programming Frameworks
  - Introduction to ML Strategy I
  - Introduction to ML Strategy II
  - Convolutional Neural Networks
  - Convolutional Neural Networks (CNN) Architectures
  - Object Detection
  - Face Recognition and Neural Style Transfer
  - Sequence Models
  - K Nearest Neighbors (KNN) Algorithm
  - Random Forest
  - Support Vector Machine
  - XGBoost

- Challenges
  - [Matrix Differentiation](https://github.com/WangCheng0116/ML/blob/main/Challenges/Matrix%20Differentiation.md)
  - [Backpropagation](https://github.com/WangCheng0116/ML/blob/main/Challenges/Back%20Propogation.md)
  - [SVM Math Proof](https://github.com/WangCheng0116/ML/blob/main/Challenges/SVM%20Math%20Proof.md)
  
- Code (code implementation for model)
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
- Interesting Resources
  - [Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html) 
  - [Neural Network Playground](https://playground.tensorflow.org/)
  - [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
  - [LaTex Cheat Sheet](https://wch.github.io/latexsheet/latexsheet.pdf)
  
  

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
|  ✅[7](##7-Oct)      | [Midterm] 8     | [Midterm] 9     | ✅[10](##10-Oct)       | ✅[11](##11-Oct)    | ✅[12](##12-Oct)  | [Midterm] 13    |
| ✅[14](##14-Oct)   | [ ] 15    | [ ] 16    | [ ] 17      | [ ] 18    | [ ] 19  | [ ] 20    |
| [ ] 21    | [ ] 22    | [ ] 23    | [ ] 24      | [ ] 25    | [ ] 26  | [ ] 27    |
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

## 14 Oct
This is a hard day. Currently I need to confess that SVM is by far the most math-heavy topic I have ever learned. On a side note, it is very interesting to see how my math courses are intertwined with SVM, because I happen to be learning non-linear optimization (MA3236). I will put SVM in [challenge](https://github.com/WangCheng0116/ML/blob/main/Challenges/SVM%20Math%20Proof.md) folder. A very simple implementation of SVM can be found [here](https://github.com/WangCheng0116/ML/blob/main/Code%20Implementation/SVM%20Binary%20Classification/SVM_Binary_Classification.ipynb) (using scikit)
