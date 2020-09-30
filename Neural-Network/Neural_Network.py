from Activation_Function import Activation_Function
from Optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
import copy

class Neural_Network:
        
    #constructor will store parameters passed from builder into nn object
    def __init__(self, X_train, Y_train, activation_function, optimizer):
        
        self.input      = X_train
        self.weights1   = np.random.rand(X_train.shape[1],7) 
        self.bias1      = np.zeros((X_train.shape[0], 7))
        self.weights2   = np.random.rand(7,1)
        self.bias2      = np.zeros((Y_train.shape[1], 1))                 
        self.y          = Y_train
        self.output     = np.zeros(Y_train.shape)
        self.optimizer = optimizer #string
        self.Optimizer = Optimizer(activation_function) #object
        self.activation_function = activation_function #string 
        self.Activation = Activation_Function()        #object

    #feedforward function
    def feedforward(self):
      
        #deal with appropiate activation
        if(self.activation_function == 'sigmoid'):
            
            self.layer1 = self.Activation.sigmoid(self.bias1 + np.dot(self.input, self.weights1))
            self.y_hat = self.Activation.sigmoid(self.bias2 + np.dot(self.layer1, self.weights2))
            error = self.loss_function(self.y_hat, self.y)
            accuracy = 1 - error

        elif(self.activation_function == 'tanh'):
        
            self.layer1 = self.Activation.tanh(self.bias1 + np.dot(self.input, self.weights1))
            self.y_hat = self.Activation.tanh(self.bias2 + np.dot(self.layer1, self.weights2))
            self.y_hat = np.absolute(self.y_hat)
            error = self.loss_function(self.y_hat, self.y)
            accuracy = 1 - error
        
        elif(self.activation_function == 'relu'): 
            
            self.layer1 = self.Activation.relu(self.bias1 + np.dot(self.input, self.weights1))
            self.y_hat = self.Activation.relu(self.bias2 + np.dot(self.layer1, self.weights2))
            #Normalizing y_hat before calculating loss
            y_hat_normalized = self.y_hat
            normalization_sum = np.linalg.norm(y_hat_normalized)
            y_hat_normalized = y_hat_normalized/ normalization_sum
            #calculating loss with regularization term
            error = self.loss_function_relu(y_hat_normalized, self.y)
            accuracy = 1 - error
            
        return error, accuracy

    #Calculating Binary Cross Entropy - USING REGULARIZATION TERM
    def loss_function_relu(self, y_hat, y):
        return float(-np.mean(y*np.log(y_hat+1e-1)))
    
    #Calculating Binary Cross Entropy
    def loss_function(self, y_hat, y):
        return float(-np.mean(y*np.log(y_hat)))
        
    def backpropagation(self):
        #updating weights, bias        
        self.weights1, self.bias1, self.weights2, self.bias2 = self.Optimizer.Adaptive_Gradient_Descent(self.activation_function, self.weights1, self.bias1, self.weights2, self.bias2, self.layer1, self.y_hat, self.y, self.input)

    #feedforward function
    def test_model(self, input_vector, y):
        
        #deal with appropiate activation
        if(self.activation_function == 'sigmoid'): 
 
            self.layer1 = self.Activation.sigmoid(np.dot(input_vector, self.weights1))
            y_hat_test = self.Activation.sigmoid(np.dot(self.layer1, self.weights2))
            accuracy = 1 - self.loss_function(y_hat_test, y)
        
        elif(self.activation_function == 'tanh'):
            
             self.layer1 = self.Activation.tanh(np.dot(input_vector, self.weights1))
             y_hat_test = self.Activation.tanh(np.dot(self.layer1, self.weights2))
             y_hat_test = np.absolute(y_hat_test)
             accuracy = 1 - self.loss_function(y_hat_test, y)

        elif(self.activation_function == 'relu'): 
            
             self.layer1 = self.Activation.relu(np.dot(input_vector, self.weights1))
             y_hat_test = self.Activation.relu(np.dot(self.layer1, self.weights2))
             accuracy = 1 - self.loss_function_relu(y_hat_test, y)

        
        return accuracy
    
