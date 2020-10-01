from Preprocessing import Preprocessing
from Neural_Network_Builder import Neural_Network_Builder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#if __name__ == "__main__":
    
###############################
# Reading dataset, preprocessing it
# and getting train and testing set
###############################
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/cervical_cancer_classification.csv'

data = Preprocessing(url)
data.Exploratory_Data_Analysis()
X_train, X_test, Y_train, Y_test = data.get_train_test_set()

###############################
# Training neural network 
# with different parameters
###############################
optimizer = "adagrad"
activation_function = ['sigmoid','tanh','relu']
epochs = [20,100, 500]
accuracy_for_different_parameters = pd.DataFrame(columns=['Testing_Accuracy','Activation_Function','Optimzer', 'Epochs'])

for activation in activation_function:
    for epoch in epochs:
        ###############################
        # TRAIN NEURAL NETWORK
        ###############################
        neural_network_builder = Neural_Network_Builder(X_train, Y_train, activation, optimizer, epoch)
        #Training the model
        training_error_list, training_accuracy_list = neural_network_builder.train_model()
        ###############################
        # Plot results from training NN
        ###############################
        plt.figure()
        plt.plot(training_error_list)
        plt.title('Error vs Epochs: ' + activation)
        plt.ylabel('error')
        plt.xlabel('iterations')
        plt.show()
        ###############################
        # TEST NEURAL NETWORK
        ###############################
        testing_accuracy = neural_network_builder.test_model(X_test, Y_test)
        new_row = {'Testing_Accuracy' : testing_accuracy, 'Activation_Function': activation, 'Optimzer': optimizer, 'Epochs': str(epoch)}
        accuracy_for_different_parameters = accuracy_for_different_parameters.append(new_row, ignore_index=True)
     
###############################
# Outputting test results from
# each NN using different parameterss
###############################
accuracy_for_different_parameters.to_csv('accuracy_for_different_parameters.csv',index= False, header= True, sep =',')






