## Information Entropy
It is first proposed by Shannon in 1948.

### Introduction
Suppose we hava a die, what is the minimum number of bits we need to represent the outcome of a roll?  
The answer is $log_2 6 \approx 2.58$ bits.  
Since it is a fair die, the probability of each outcome is the same, and it has 6 possible outcomes. So we need $log_2 6$ bits to represent the outcome of a roll.

But what if the die is not fair?  
We need to first convert the outcome into an event has equal probability. 
> For example, if the chance of winning a lottery is $\frac{1}{2000}$, then we can treat it as picking a particular ball from a bag of 2000 balls.  

Let's say $P(H) = 0.8$ and $P(T) = 0.2$, then for $H$, we need $log_2 \frac{1}{0.8} \approx 0.32$ bits, and for $T$, we need $log_2 \frac{1}{0.2} \approx 2.32$ bits.  
But since $H$ and $T$ have different probabilities, we take the expectation of it.

## Formal Definition
Suppose we have a discrete random variable $X$ with probability mass function $p(x)$, then the entropy of $X$ is defined as
$$
H(X) = - \sum_{x \in X} p(x) log_2 p(x)
$$
### Summary
Information Entropy gives the limit of compression of any information source.  
The perspective of understanding the formula is 
$$
H(X) = \sum_{x \in X} p(x) 
log_2 \frac{1}{p(x)}
$$
It has three steps:
- $\frac{1}{p(x)}$ is the measurement of uncertainty level
- $log_2\frac{1}{p(x)}$ is the conversion of uncertainty level to bits
- $p(x)log_2\frac{1}{p(x)}$ is the expectation

## Cross Entropy
### Formal Definition
$$H(y, \hat{y}) = - \sum_{i=1}^n y_i log_2 \hat{y}_i$$
One natural question, can we swap the position of $y$ and $\hat{y}$?  
The answer is no. In last section, $\frac{1}{p(x)}$ is the measurement of uncertainty level, and naturally we want to calculate the uncertainty level of $\hat{y}$.
> Cross Entropy basically calculates the difference between two distributions.

### Relation between Cross Entropy and Information Entropy
$$
 - \sum_{i=1}^n y_i log_2 \hat{y}_i \geq - \sum_{i=1}^n y_i log_2 y_i
 $$
 i.e. Cross Entropy is always greater than or equal to Information Entropy.  

 **Proof:**
since
$$
H(p, q) = D_{KL}(p||q) + H(p)
$$
and $D_{KL}(p||q) \geq 0$,  so $H(p, q) \geq H(p)$.

## KL Divergence
### Intuition
Suppose we have two coins, $A$ and $B$, and we want to know how different they are.  
Let's denote:
$$
\begin{aligned}
P(A=H) &= p_1 \\
P(A=T) &= p_2 \\
P(B=H) &= q_1 \\
P(B=T) &= q_2
\end{aligned}
$$
Then is there anyway we can quantify the difference between $A$ and $B$?  
The approach is we first generate a sequence of observations and then see the difference between the two sequences.  

Suppose we have $N$ observations where $H$ appears $n_H$ times and $T$ appears $n_T$ times, then
$$
\begin{aligned}
\frac{P(observation|A)}{P(observation|B)} &= \frac{p_1^{n_H}p_2^{n_T}}{q_1^{n_H}q_2^{n_T}} \\
\end{aligned}
$$
By normalizing and taking $log$ of it, we get
$$
\begin{aligned}
log(\frac{p_1^{n_H}p_2^{n_T}}{q_1^{n_H}q_2^{n_T}})^{\frac{1}{N}} &= \frac{N_H}{N}logp_1 + \frac{N_T}{N}logp_2 - \frac{N_H}{N}logq_1 - \frac{N_T}{N}logq_2 \\
\end{aligned}
$$
Let $N$ goes to infinity, we get
$$
\begin{aligned}
\lim_{N \to \infty} \frac{N_H}{N}logp_1 + \frac{N_T}{N}logp_2 - \frac{N_H}{N}logq_1 - \frac{N_T}{N}logq_2 \\= p_1logp_1 + p_2logp_2 - p_1logq_1 - p_2logq_2 \\
= p_1log\frac{p_1}{q_1} + p_2log\frac{p_2}{q_2} \\
\end{aligned}
$$
This is the KL Divergence between $A$ and $B$.

### Formal Definition
$$
D_{KL}(p||q) = \sum_{i=1}^n p_i log \frac{p_i}{q_i}
$$
where $P$ is the true distribution and $Q$ is the estimated distribution.
### Properties
* KL Divergence is not symmetric, i.e. $D_{KL}(p||q) \neq D_{KL}(q||p)$
* As long as there is small difference between $P$ and $Q$, $D_{KL}(p||q)$ will be greater than 0.

**Proof:**
$$
\begin{aligned}
-D_{KL}(p||q) &= 
\sum_{i=1}^n p_i log \frac{q_i}{p_i} \\
&\leq log \sum_{i=1}^n p_i \frac{q_i}{p_i} \\
&= log \sum_{i=1}^n q_i \\
&= 0
\end{aligned}
$$
since $log$ is a concave function

### Relation between KL Divergence and Cross Entropy
$$
\begin{aligned}
D_{KL}(p||q) &= \sum_{i=1}^n p_i log \frac{p_i}{q_i} \\
&= -\sum_{i=1}^n p_i log \frac{1}{p_i} + \sum_{i=1}^n p_i log \frac{1}{q_i} \\
&= -H(p) + H(p, q)
\end{aligned}
$$
> Since $H(p)$ is a constant, meaning that in ML, minimizing the Cross Entropy is equivalent to minimizing the KL Divergence.


# Summary
* Information Entropy:
$$
H(x) = - \sum_{x \in X} p(x) log_2 p(x) \\
= \sum_{x \in X} p(x) log_2 \frac{1}{p(x)}
$$
> Its' perfect encoding.
* Cross Entropy:
$$
H(p, q) = - \sum_{i=1}^n p_i log_2 q_i \\
$$
> It's imperfect encoding.
* KL Divergence:
$$
D_{KL}(p||q) = \sum_{i=1}^n p_i log \frac{p_i}{q_i}
$$
* Relation
$$
\begin{aligned}
 D_{KL}(p||q) =  H(p, q) -H(p) \\
\end{aligned}
$$
> Personal thought: KL-Divergence is the difference between two distributions without considering true distribution(since it's know already).
---
Reference  
[Youtube](https://www.youtube.com/watch?v=SxGYPqCgJWM)