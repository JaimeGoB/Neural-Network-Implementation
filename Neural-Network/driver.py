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
activation_function = ["sigmoid"]
epochs = [200]

for activation in activation_function:
    for epoch in epochs:
        neural_network_builder = Neural_Network_Builder(X_train, Y_train, activation, optimizer, epoch)
        training_accuracy = neural_network_builder.train_model()
        
print('training error')
training_accuracy[199]
print('test error')
neural_network_builder.test_model(X_test, Y_test)




# X = np.array([[0,0,1],
#               [0,1,1],
#               [1,0,1],
#               [1,1,1]])
# len(X)
# y = np.array([[0],[1],[1],[0]])
# nn = NeuralNetwork(X,y)

