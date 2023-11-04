# Loss Function
Suppose we have content image $C$ and style image $S$, and we want to generate a new image $G$ that has the content of $C$ and the style of $S$. We can define the content loss and style loss as follows:
$$
\begin{aligned}
\mathcal{L}_{content}(\hat{C}, \hat{G}) &= \frac{1}{2} \sum_{i,j} (F_{ij}^C - F_{ij}^G)^2 \\
\mathcal{L}_{style}(\hat{S}, \hat{G}) &= \sum_{l=0}^{L} \lambda_l \sum_{i,j} (G_{ij}^l - S_{ij}^l)^2
\end{aligned}
$$
where $\hat{C}$, $\hat{S}$, and $\hat{G}$ are the normalized versions of $C$, $S$, and $G$, respectively. $F_{ij}^C$ and $F_{ij}^G$ are the feature maps of $C$ and $G$ at layer $i$ and position $j$. $G_{ij}^l$ and $S_{ij}^l$ are the Gram matrices of $G$ and $S$ at layer $l$. $\lambda_l$ is the weight of the style loss at layer $l$. The total loss is defined as:
$$
\mathcal{L}_{total} = \alpha \mathcal{L}_{content} + \beta \mathcal{L}_{style}
$$
where $\alpha$ and $\beta$ are the weights of the content loss and style loss, respectively.