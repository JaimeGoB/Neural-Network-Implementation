import numpy as np
import pandas as pd

class Activation_Function:
    
    def sigmoid(self, x):
        return 1.0/(1+ np.exp(-x))
   
    def sigmoid_derivative(self, x):
        return x * (1.0 - x)
