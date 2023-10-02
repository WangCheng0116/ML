Another detailed illustration of back propogation - [sourece](https://zhuanlan.zhihu.com/p/71892752)  
The key points are how to find the recursive relation bewtween $\frac{\partial Cost}{\partial z^{[l]}}$ and $\frac{\partial Cost}{\partial z^{[l-1]}}$, or equivalently $\delta^{[l]}$ and $\delta^{[l-1]}$

It is good to revise this source from time to time in case I get confused in the future.

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

