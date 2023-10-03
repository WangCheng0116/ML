# Machine Learning Daily Blog  
## Overview  
- Basic Concepts ([notes](https://github.com/WangCheng0116/ML/blob/main/AndrewNgMLCourse/note.md))
  - Linear Regression with one variable
  - Linear Regression with multiple variables
  - Features and Polynomial Regression
  - Logistic Regression 
  - Regularization
  - Neural Networks
  - Deep Neural Network
  - Setting Up ML Application
  - Optimization Algorithms
  - Hyperparameter Tuning Batch Normalization and Programming Frameworks
  - Introduction to ML Strategy I
  - Introduction to ML Strategy II
- Projects (code implementation for model)
  - [Triple layers NN for digit recognition (without frameworks)](https://github.com/WangCheng0116/ML/tree/main/Projects/Digit%20Recognition)
  - [Linear Regression](https://github.com/WangCheng0116/ML/tree/main/Projects/Linear%20Regression)
  - Decision Tree
- Highlights and challenges
  - [Gradient Descent in Back Propagation](https://github.com/WangCheng0116/ML/blob/main/Challenges/Back%20Propogation.md) (Why? Both rigorous deduction and intuitive understanding)
  - [Matrix Differentiation](https://github.com/WangCheng0116/ML/blob/main/Challenges/Matrix%20Differentiation.md)
- Tools
  - [numpy](https://numpy.org/doc/stable/reference/) Notice: axis = 0, do things vertically; axis = 1, do things horizontally (even '1' is vertical)

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
|           | ✅ [1](##1-Oct)     |  ✅ [2](##2-Oct)     |  [ ] 3       | [ ] 4     | [ ] 5   | [ ] 6     |
| [ ] 7     | [ ] 8     | [ ] 9     | [ ] 10      | [ ] 11    | [ ] 12  | [ ] 13    |
| [ ] 14    | [ ] 15    | [ ] 16    | [ ] 17      | [ ] 18    | [ ] 19  | [ ] 20    |
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
## 4 Oct
[DNN](https://github.com/WangCheng0116/ML/tree/main/Projects/Deep%20Neural%20Network)
