1. Notation
$$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$$
$$a^{(l)}=\sigma(z^{(l)})$$
2. Cost Function
Suppose the cost function is MSE, i.e. 
$$C(W, b) = \frac{1}{2}||a^{(l)} - \hat{y}||^2$$
3. Gradient at output layer ($l$)
$$\begin{aligned}
&\begin{aligned}
\frac{\partial C(W, b)}{\partial w^{(l)}} & =\frac{\partial C(W, b)}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial w^{(l)}} \\
& =\left(a^{(l)}-y\right) \odot \sigma^{\prime}\left(z^{(l)}\right) a^{(l-1)}

\end{aligned}\\
&\begin{aligned}
\frac{\partial C(W, b)}{\partial b^{(l)}} & =\frac{\partial C(W, b)}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}} \frac{\partial z^{(l)}}{\partial b^{(l)}} \\
& =\left(a^{(l)}-y\right) \odot \sigma^{\prime}\left(z^{(l)}\right)
\end{aligned}
\end{aligned}$$
> The reason why we have $a^{(l)}-y$ is because the cost function is MSE. If the cost function is cross entropy, then we have $\frac{\partial C(W, b)}{\partial a^{(l)}} = \frac{a^{(l)}-y}{a^{(l)}(1-a^{(l)})}$  
  
> Dimension of each term:
> $$dim(\frac{\partial C(W, b)}{\partial a^{(l)}} )= dim(a^{(l)}): (n_y, m)$$
> $$dim(\frac{\partial a^{(l)}}{\partial z^{(l)}} )= dim(z^{(l)}): (n_y, m)$$
> $$dim(\frac{\partial z^{(l)}}{\partial w^{(l)}} )= dim(w^{(l)}): (n_y, n_{l-1})$$

4. Gradient at hidden layers
Notice that 
$$
\delta^{(l)}=\frac{\partial C(W, b)}{\partial a^{(l)}} \frac{\partial a^{(l)}}{\partial z^{(l)}}=\left(a^{(l)}-y\right) \odot \sigma^{\prime}\left(z^{(l)}\right)
$$  
is a repeated term, so we can define it as $\delta^{(l)}$ for convenience.  

$$
\delta^{(l)}=\frac{\partial C(W, b)}{\partial z^{(l+1)}} \frac{\partial z^{(l+1)}}{\partial z^{(l)}}=\delta^{(l+1)} \frac{\partial z^{(l+1)}}{\partial z^{(l)}}
$$
Notice that $z^{(l+1)}$ and $z^{(l)}$ satisfies：
$$z^{(l+1)}=W^{(l+1)}a^{(l)}+b^{(l+1)}=W^{(l+1)}\sigma(z^{(l)})+b^{(l+1)}$$
so,
$$\frac{\partial z^{(l+1)}}{\partial z^{(l)}}=\left(W^{(l+1)}\right)^T \odot \sigma^{\prime}\left(z^{(l)}\right)$$
so the recursive relation is: 
$$
\delta^{(l)}=\left(W^{(l+1)}\right)^T \delta^{(l+1)} \odot \sigma^{\prime}\left(z^{(l)}\right)
$$
Then we can get the gradient at hidden layers:
$$
\begin{aligned}
& \frac{\partial C(W, b)}{\partial w^{(l)}}=\delta^{(l)} a^{(l-1)} \\
& \frac{\partial C(W, b)}{\partial b^{(l)}}=\delta^{(l)}
\end{aligned}
$$
credits to [source](https://zhuanlan.zhihu.com/p/71892752)  
The key points are how to find the recursive relation bewtween $\frac{\partial Cost}{\partial z^{[l]}}$ and $\frac{\partial Cost}{\partial z^{[l-1]}}$, or equivalently $\delta^{[l]}$ and $\delta^{[l-1]}$

Following is illustration with diagrams:

![Alt text](IMG_C04A1ACDA7E9-1.jpeg)
![Alt text](IMG_DB11AE717A55-1.jpeg)
