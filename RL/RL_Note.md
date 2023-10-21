<!-- toc -->
# Basic Concepts
* State: $s_t$ describes the agent's status with respect to the environment.
* State Space: $S$ is the set of all possible states.
* Action: $a_t$ is the action taken by the agent at time $t$.
* Action Space: $A$ is the set of all possible actions.
* State Transition: A process where the agent moves from one state to another.
* State Transition Probability: $p(s'|s,a)$ is the probability of transitioning from state $s$ to state $s'$ under action $a$.
* Policy: tells what action to take at a state. $\pi(a|s)$, which means the probability of taking action $a$ at state $s$.
* Reward: 
* Reward Probability: $p(r|s,a)$ is the probability of getting reward $r$ at state $s$ under action $a$.
* Trajectory: is state-action-reward sequence. 
* Return: summation of all rewards in a trajectory. 
* Discount Factor: $\gamma \in (0,1)$ is the discount factor. It is used to discount the future reward.

# Bellman Equation
## Motivating Example
![All](image.png)  

Suppose $v_i$ is the return value of a trajectory starting at state $s_i$, then we have (consider discount factor $\gamma$):
$$
\begin{align}
v_1 &= r_1 + \gamma r_2 + \gamma^2 r_3 + ... \\
v_2 &= r_2 + \gamma r_3 + \gamma^2 r_4 + ... \\
v_3 &= r_3 + \gamma r_4 + \gamma^2 r_1 + ... \\
v_4 &= r_4 + \gamma r_1 + \gamma^2 r_2 + ... \\
\end{align}
$$
which is equivalent to:
$$
\begin{align}
v_1 &= r_1 + \gamma v_2 \\
v_2 &= r_2 + \gamma v_3 \\
v_3 &= r_3 + \gamma v_4 \\
v_4 &= r_4 + \gamma v_1 \\
\end{align}
$$
In matrix form:
$$
\begin{bmatrix}
v_1 \\
v_2 \\
v_3 \\
v_4 \\
\end{bmatrix}

=
\begin{bmatrix}
    r_1 \\
    r_2 \\
    r_3 \\
    r_4 \\
\end{bmatrix}
+ 
\gamma
\begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    1& 0 & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
v_1 \\
v_2 \\
v_3 \\
v_4 \\
\end{bmatrix}
$$
Then 
$$v\,=\,(I\,-\,\gamma P)^{-1}r$$
## State Value 
Consider the following single step 
$$\textstyle S_{t}\ {\overset{A_{t}}{\longrightarrow}}\ S_{t+1},R_{t+1}$$
where $S_t$ is the state at time $t$, $A_t$ is the action taken at time $t$, $S_{t+1}$ is the state at time $t+1$, and $R_{t+1}$ is the reward at time $t+1$.
> $S_t$, $S_{t+1}$, $A_{t}$ and $R_{t+1}$ are random variables.

* Policy: $S_t \rightarrow A_{t}$, $\pi(A_t = a|S_t = s)$
* Reward: $S_t, A_t \rightarrow R_{t+1}$, $p(R_{t+1} = r|S_t = s, A_t = a)$
* State Transition: $S_t, A_t \rightarrow S_{t+1}$, $p(S_{t+1} = s'|S_t = s, A_t = a)$

Then we can have multi-step trajectory:
$$S_{t}\stackrel{A t}{\longrightarrow}S_{t+1},\,R_{t+1}\stackrel{A_{t+1}}{\longrightarrow}S_{t+2},\,{{{R_{t+2}}}}\stackrel{A_{t+2}}{\longrightarrow}S_{t+3},\,R_{t+3}\cdot\cdot\cdot\cdot.$$

The return of this trajectory is:
$$G_{t}= R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\cdot\cdot\cdot\cdot=\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1}$$

So the **state value** function is defined as the expected return starting from state $s$:
$$v_{\pi}(s)=\mathbb{E}[G_{t}|S_{t}=s]$$
> The difference between state value and return is that state value is an expected sum of multiple trajectories, while return is the sum of a single trajectory.

## Bellman Equation 
Back to the example above, we have:

multi-step trajectory:
$$S_{t}\stackrel{A t}{\longrightarrow}S_{t+1},\,R_{t+1}\stackrel{A_{t+1}}{\longrightarrow}S_{t+2},\,{{{R_{t+2}}}}\stackrel{A_{t+2}}{\longrightarrow}S_{t+3},\,R_{t+3}\cdot\cdot\cdot\cdot.$$

The return of this trajectory is:

$$G_{t}= R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\cdot\cdot\cdot=\sum_{k=0}^{\infty}\gamma^{k} R_{t+k+1} \\
\Leftrightarrow G_{t}=R_{t+1}+\gamma G_{t+1}$$


So
$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[G_{t}|S_{t}=s] \\
&= \mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_{t}=s] \\
&= \mathbb{E}[R_{t+1}|S_{t}=s]+\gamma \mathbb{E}[G_{t+1}|S_{t}=s]
\end{align}
$$
The first term is
$$
\begin{align}
\mathbb{E}[R_{t+1}|S_{t}=s] 

    &= \sum_{a\in\mathcal{A}}\pi(a|s)\mathbb{E}[R_{t+1}|S_{t}=s,A_{t}=a] ~~~~~~~~(Total ~Probability) 
    \\
    &= \sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}r p(r|s,a) ~~~~~~~~(Total ~Expectation)
\end{align}
$$
> This term is the called expectation of **immediate reward**.

The second term is
$$
\begin{align}
\mathbb{E}[G_{t+1}|S_{t}=s] 

  &= \sum_{s^{\prime}\in{\cal S}}\mathbb{R}[G_{t+1}|S_{t}=s,S_{t+1}= s^{\prime}]p(s^{\prime}|s) ~~(Total ~Probability)
    \\
    &= \sum_{s^{\prime}\in{\cal S}}\mathbb{E}[G_{t+1}|S_{t+1}=s^{\prime}]p(s^{\prime}|s) ~~ (Markov ~ Property  -Memoryless)
    \\
    &= \sum_{s^{\prime}\in{\cal S}}v_{\pi}(s^{\prime})p(s^{\prime}|s)
    \\
    &=\sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})\sum_{a\in A}p(s^{\prime}|s,a)\pi(a|s)
    ~~ (Total ~ Probability)
    \\
    &= \sum_{a\in\mathcal{A}}\pi(a|s)\sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})p(s^{\prime}|s,a)
\end{align}
$$
> This term is the called expectation of **future reward**.

So Bellman Equation is:
$$
\begin{align}
v_{\pi}(s) &= \mathbb{E}[R_{t+1}|S_{t}=s]+\gamma \mathbb{E}[G_{t+1}|S_{t}=s] \\
&= \sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}r p(r|s,a) + \gamma \sum_{a\in\mathcal{A}}\pi(a|s)\sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})p(s^{\prime}|s,a) \\
&= \sum_{a\in\mathcal{A}}\pi(a|s)\left[\sum_{r\in\mathcal{R}}r p(r|s,a) + \gamma \sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})p(s^{\prime}|s,a)\right] \\
\end{align}
$$
> It is important to note that how Bellman equation can connect the state value of current state $v_{\pi}(s)$ with the state value of all next possible state $v_{\pi}(s^{\prime})$.

- In English
  - Outer Sigma: From all possible actions, choose action $a$
    - First Inner Sigma: From all possible rewards, choose reward $r$ and calculate the probability of getting that. (Essentially is expected reward)
    - Second Inner Sigma: From all possible next states, choose state $s^{\prime}$ and calculate the probability of getting there. Multiply with the state value of that state $s'$. (Essentially is expected future reward)

## Bellman Equation in Matrix Form
Previously, we know 
$$
\begin{align}
v_{\pi}(s)
&= \sum_{a\in\mathcal{A}}\pi(a|s)\sum_{r\in\mathcal{R}}r p(r|s,a) + \gamma\sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})\sum_{a\in A}p(s^{\prime}|s,a)\pi(a|s)
\end{align}
$$

which can be simplified as:
$$v_{\pi}(s)=r_{\pi}(s)+\gamma\sum_{s^{\prime}\in S}p_{\pi}(s^{\prime}|s)v_{\pi}(s^{\prime})$$
where
$$
r_{\pi}(s)=\sum_{a\in A}\pi(a|s)\sum_{r\in R}p(r|s,a)r \\
p_{\pi}(s^{\prime}|s)=\sum_{a\in A}\pi(a|s)p(s^{\prime}|s,a)$$
> $r_{\pi}(s)$ is the expected immediate reward of state $s$ under policy $\pi$.
> $p_{\pi}(s^{\prime}|s)$ is the probability of transitioning from state $s$ to state $s'$ under policy $\pi$.

Suppose state has $n$ states, then we can have:
$$
\begin{align}
v_{\pi}(s_i) = r_{\pi}(s_i)+\gamma\sum_{j=1}^{n}p_{\pi}(s_j|s_i)v_{\pi}(s_j)
\end{align}
$$
In matrix form:

$$
\begin{align}
v_{\pi} &= r_{\pi}+\gamma P_{\pi}v_{\pi} \\
\end{align}
$$
where 
$$
\begin{align}
v_{\pi} &= \begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
\vdots \\
v_{\pi}(s_n) \\
\end{bmatrix}
\\
r_{\pi} &= \begin{bmatrix}
r_{\pi}(s_1) \\
r_{\pi}(s_2) \\
\vdots \\
r_{\pi}(s_n) \\
\end{bmatrix}
\\
P_{\pi} &= \begin{bmatrix}
p_{\pi}(s_1|s_1) & p_{\pi}(s_2|s_1) & \cdots & p_{\pi}(s_n|s_1) \\
p_{\pi}(s_1|s_2) & p_{\pi}(s_2|s_2) & \cdots & p_{\pi}(s_n|s_2) \\
\vdots & \vdots & \ddots & \vdots \\
p_{\pi}(s_1|s_n) & p_{\pi}(s_2|s_n) & \cdots & p_{\pi}(s_n|s_n) \\
\end{bmatrix}
\end{align}
$$
> $P_{\pi}$ is called row stochastic matrix, which means the sum of each row is 1.

> $(i, j)$-entry of $P_{\pi}$ is the probability of transitioning from state $s_i$ to state $s_j$ under policy $\pi$.

Let's look at an example:
![Alt text](image-1.png)
$$
\begin{align}
\begin{bmatrix}
   v_{\pi}(s_1) \\
    v_{\pi}(s_2) \\
    v_{\pi}(s_3) \\
    v_{\pi}(s_4) \\ 
\end{bmatrix}
&=
\begin{bmatrix}
0.5 * 0 + 0.5 * (-1) \\
1 \\
1 \\
1 \\
\end{bmatrix}+
\gamma
\begin{bmatrix}
0 & 0.5 & 0.5 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
v_{\pi}(s_3) \\
v_{\pi}(s_4) \\
\end{bmatrix}
\end{align}
$$
> (1,3)-entry of $P_{\pi}$ is 0.5, which means the probability of transitioning from state $s_1$ to state $s_3$ under policy $\pi$ is 0.5.

## Solving Bellman Equation (Closed Form and Iterative Method)
* Closed Form

We know
$$v_{\pi} = r_{\pi}+\gamma P_{\pi}v_{\pi}$$
So
$$v_{\pi} = (I - \gamma P_{\pi})^{-1}r_{\pi}$$
> This method is not efficient (inversion is $O(n^3)$).

---
**Why $I - \gamma P_{\pi}$ is invertible?**  

If $(I - \gamma P_{\pi}) x = 0$, then $Ax = \frac{1}{\gamma} x$.
Now suppose that $x \neq 0$, then $\frac{1}{\gamma}$ is an eigenvalue of $P_{\pi}$.
Since $P_{\pi}$ is a row stochastic matrix, $\frac{1}{\gamma} \leq 1$, a contradiction.
Thus $x = 0$.
Hence $I - \gamma P_{\pi}$ is invertible.  

---
**Why eigenvalue of a stochastic matrix $A$ is less than 1?**

Let $\lambda$ be an eigenvalue of the stochastic matrix $A$, and let $v$ be a corresponding eigenvector. That is, we have $Av = \lambda v$.  
Comparing the $i$-th row of both sides, we obtain:

$$
a_{i1}v_1 + a_{i2}v_2 + \ldots + a_{in}v_n = \lambda v_i \quad (*)
$$

for $i=1, \ldots, n$. 

Let $|v_k| = \max\{|v_1|, |v_2|, \ldots, |v_n|\}$; namely, $v_k$ is the entry of $v$ that has the maximal absolute value. 

Note that $|v_k| > 0$ since otherwise we would have $v = 0$, which contradicts the fact that an eigenvector is a nonzero vector. Then, from (*) with $i=k$, we have:

$$
|\lambda| \cdot |v_k| = |a_{k1}v_1 + a_{k2}v_2 + \ldots + a_{kn}v_n| \leq a_{k1}|v_1| + a_{k2}|v_2| + \ldots + a_{kn}|v_n| \leq a_{k1}|v_k| + a_{k2}|v_k| + \ldots + a_{kn}|v_k| = (a_{k1} + a_{k2} + \ldots + a_{kn}) |v_k| = |v_k|.
$$

This follows from the triangle inequality and the fact that $a_{ij} \geq 0$.



* Iterative Method
$$v_{k+1}=r_{\pi}+\gamma P_{\pi}v_{k},\quad k=0,1,2,\cdot\cdot\cdot$$
when $k$ is large enough, $v_k$ will converge to $v_{\pi}$.

---
Proof:
Define the error as:
$$e_{k}=v_{k}-v_{\pi}$$
Then we want to show that $e_k$ converges to 0 as $k$ goes to infinity.
$$
\begin{align}
e_{k+1} &= v_{k+1}-v_{\pi} \\
&= r_{\pi}+\gamma P_{\pi}v_{k}-v_{\pi} \\
&= r_{\pi}+\gamma P_{\pi}v_{k}-r_{\pi}-\gamma P_{\pi}v_{\pi} \\
&= \gamma P_{\pi}(v_{k}-v_{\pi}) \\
&= \gamma P_{\pi}e_{k}
\end{align}
$$
So
$$e_{k+1} = \gamma P_{\pi}e_{k} = \gamma P_{\pi}(\gamma P_{\pi}e_{k-1}) = \gamma^2 P_{\pi}^2 e_{k-1} = \cdots = \gamma^{k+1} P_{\pi}^{k+1} e_{0}$$
Every entry of $P_{\pi}$ is trivially between 0 and 1, so $P_{\pi}^{k+1}$ is also between 0 and 1. Plus, $\gamma \in (0,1)$, so $\gamma^{k+1} P_{\pi}^{k+1}$ converges to 0 as $k$ goes to infinity. So $e_{k+1}$ converges to 0 as $k$ goes to infinity.

## Action Value
Let's first compare state value and action value:
* State Value: $v_{\pi}(s)$ is the expected return starting from state $s$.
* Action Value: $q_{\pi}(s,a) = \mathbb{E}[G_{t}|S_{t}=s,A_{t}=a]$ is the expected return starting from state $s$ and taking action $a$.
The relationship between state value and action value is:
$$v_{\pi}(s) = \sum_{a\in\mathcal{A}}\pi(a|s)q_{\pi}(s,a)
\quad {(a)}
$$
> In english: state value is the weighted sum of action value, where the weight is the probability of taking that action.
We notice that 
$$
v_{\pi}(s) = \sum_{a\in\mathcal{A}}\pi(a|s)\left[\sum_{r\in\mathcal{R}}r p(r|s,a) + \gamma \sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})p(s^{\prime}|s,a)\right]
\quad {(b)}
$$
By comparing (a) and (b), we can see 
$$
\begin{align}
q_{\pi}(s,a) &= \sum_{r\in\mathcal{R}}r p(r|s,a) + \gamma \sum_{s^{\prime}\in{\mathcal{S}}}v_{\pi}(s^{\prime})p(s^{\prime}|s,a) \\

\end{align}
$$
> In english: We first fix an action $a$, then we sum over all possible next states $s^{\prime}$, and for each $s^{\prime}$, we multiply the state value of $s^{\prime}$ with the probability of transitioning from $s$ to $s^{\prime}$ under action $a$. And last, we add it with expected immediate reward.  
---
Example:
![Alt text](image-2.png)
$$
\begin{align}
q_{\pi}(s_1, a_2) &= -1 + \gamma v_{\pi}(s_2) \\
q_{\pi}(s_1, a_3) &= 0 + \gamma v_{\pi}(s_3) \\
\end{align}
$$
> Note that we don't need to multiply $0.5$ since action has been fixed.

## Summary
**Key Concepts and Results:**


* State Value:
The state value, denoted as $v_\pi(s)$, is defined as the expected return $E[G_t | S_t = s]$.

* Action Value:
The action value, denoted as $q_\pi(s, a)$, is defined as the expected return $E[G_t | S_t = s, A_t = a]$.

* Bellman Equation (Elementwise Form):
The Bellman equation in its elementwise form is:
$$
v_\pi(s) = \sum_a \pi(a|s) \left(\sum_r p(r|s, a)r + \gamma \sum_{s'} p(s'|s, a)v_\pi(s')\right)
$$

* Bellman Equation (Matrix-Vector Form):
The Bellman equation in its matrix-vector form is:
$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$
* Solving the Bellman Equation:
Using a closed-form solution or an iterative solution.
