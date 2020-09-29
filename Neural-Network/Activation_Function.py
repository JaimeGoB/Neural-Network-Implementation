import numpy as np
import pandas as pd

class Activation_Function:
    
    def sigmoid(self, x):
        return 1.0/(1+ np.exp(-x))
   
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def tanh(self,x):
        return np.tanh(x)
    
    def tanh_derivative(self,x):
        return 1.0 - np.tanh(x)**2
    

    def relu(self,x):
        return np.minimum(1.0, x)
    
    def relu_derivative(self,x):
        x[x<=0] = 0
        x[x>0] = 1
        return x
