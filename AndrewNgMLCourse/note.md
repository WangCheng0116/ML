# Linear Regression with one variable  
Our hypothesis model:  
![image](https://github.com/WangCheng0116/ML/assets/111694270/c71fb25e-86cb-4792-a6f0-41cd991fe4b1)  
the function to estimate our errors between real data and our model  
![image](https://github.com/WangCheng0116/ML/assets/111694270/893c1a2c-55de-4404-9524-09e58f1a80ea)  
This is the diagram of lost function in a 3d space  
<img width="181" alt="image" src="https://github.com/WangCheng0116/ML/assets/111694270/9ddb83aa-4fdf-4b60-9baa-3c50ee5aec8d">  
## Gradient Descent  
The idea behind gradient descent is as follows: Initially, we randomly select a combination of parameters (θ₀, θ₁, ..., θₙ), calculate the cost function, and then look for the next combination of parameters that will cause the cost function to decrease the most. We continue to do this until we reach a local minimum because we haven't tried all possible parameter combinations, so we cannot be sure if the local minimum we find is the global minimum. Choosing different initial parameter combinations may lead to different local minima.

