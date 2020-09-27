from Activation_Function import Activation_Function
import numpy as np
import pandas as pd


class Optimizer:
    def __init__(self):
        self.Activation = Activation_Function() #object

        
    def Adaptive_Gradient_Descent(self,activation_function, weights1, bias1, weights2, bias2, layer1, y_hat, y, input, learning_rate = 1e-2, epsilon = 1e-6):

        #################################
        ## Computing gradients needed for 
        ## Adaptive Gradient Descent
        #################################
        if(activation_function == 'sigmoid'): 
            #Computing gradient/derivative of weights2 with sigmoid derivative
            derivative_weights2 = np.dot(layer1.T, (2*(y - y_hat) * self.Activation.sigmoid_derivative(y_hat)))
            derivative_weights1 = np.dot(input.T,  (np.dot(2*(y - y_hat) * self.Activation.sigmoid_derivative(y_hat), weights2.T) * self.Activation.sigmoid_derivative(layer1)))
        # else if(self.activation_function == 'tanh'): 
        # else if(self.activation_function == 'relu'): 
            
        #################################
        ## UPDATE THE WEIGHTS IN
        ## INPUT LAYER AND HIDDEN LAYER
        ## USING ADAPTIVE GRADIENT DESCENT
        #################################
        
        #################################
        ## Updating hidden layer weights
        #################################
        #Squaring gradient and adding it up
        Sum_Gradient_Squared = np.sum(derivative_weights2.T **2)
        
        #Calculating gradient update term for AdaGrad Equation
        gradient_update = derivative_weights2 / (np.sqrt(Sum_Gradient_Squared + epsilon))    
        
        #updating weights based on adaptive gradient descent
        weights2 = weights2 - (learning_rate * gradient_update)
        
        #################################
        ## Updating bias2
        #################################
        bias2 += np.sum(-learning_rate * derivative_weights2)

        #################################
        ## Updating input layer weights
        #################################

        #Squaring gradient and adding it up
        Sum_Gradient_Squared = np.sum(derivative_weights1.T **2)
        
        #Calculating gradient update term for AdaGrad Equation
        gradient_update = derivative_weights1 / (np.sqrt(Sum_Gradient_Squared + epsilon))    
         
        # update the weights with the derivative (slope) of the loss function
        weights1 = weights1 - (learning_rate * gradient_update)
        
        #################################
        ## Updating bias1
        #################################
        bias1 += np.sum(-learning_rate * derivative_weights1)

        return weights1, bias1, weights2,  bias2