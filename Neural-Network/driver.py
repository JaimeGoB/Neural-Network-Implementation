from Preprocessing import Preprocessing
from Neural_Network_Builder import Neural_Network_Builder
import numpy as np


#if __name__ == "__main__":
    
#Setting up URL to read dataset
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/cervical_cancer_classification.csv'

data = Preprocessing(url)
data.Exploratory_Data_Analysis()
X_train, X_test, Y_train, Y_test = data.get_train_test_set()

optimizer = "adagrad"
activation_function = ['sigmoid', 'tanh']
epochs = [200]

for activation in activation_function:
    for epoch in epochs:
        #Create NN handler to manage NN
        neural_network_builder = Neural_Network_Builder(X_train, Y_train, activation, optimizer, epoch)
        #Training the model
        training_accuracy_list = neural_network_builder.train_model()
        #getting accuracy
        training_accuracy = training_accuracy_list[-1]
        #getting accuracy for testing set
        testing_accuracy, y_hat_test = neural_network_builder.test_model(X_test, Y_test)
        print('Accuracy ' + activation)
        print(testing_accuracy)
        
###############################
# Confusion Matrix Here
###############################





