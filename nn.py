import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#X,y = sklearn.datasets.make_moons(200, noise = 0.15)

#Setting up URL to read dataset
url = 'https://raw.githubusercontent.com/JaimeGoB/Data-Sets/master/cervical_cancer_classification.csv'

class Preprocessing:
    
    #setting up empty dataframe for dataset
    data = pd.DataFrame()
    
    def __init__(self,url):
        #reading file from url
        self.data = pd.read_csv(url)
        
        #converting data types to float to perform calculations on NN
        self.data.astype('float64').dtypes

        print(self.data.dtypes)
        
     
    def Exploratory_Data_Analysis(self):

        #checking for missing values or NaN
        self.data.isna().sum()
    
        # #Checking if dependent variable has a linear relationship with the attributes.
        # sns.set(rc={'figure.figsize':(11.7,8.27)})
        # sns.swarmplot(x=self.data['ca_cervix'], y=self.data['motivation_willingness'], color=".25").set_title('Linearity Check')
        # sns.boxplot(self.data['ca_cervix'], self.data['motivation_willingness'])
        # plt.show()
        
        #Checking correlation between predictors 
        #We will use Emerging Markets Index.
        correlation_matrix = self.data.corr().round(2)
        #annot = True to print the values inside the square
        #sns.heatmap(data=correlation_matrix, annot=True).set_title('Heat Map')

        #Droping predictors not needed. The model will only use:
        # motivation_willingness
        # socialSupport_emotionality
        # socialSupport_appreciation 
        # socialSupport_instrumental 
        # empowerment_knowledge    
        # empowerment_abilities     
        # empowerment_desires
        self.data.drop(['behavior_sexualRisk'], axis=1, inplace = True)
        self.data.drop(['behavior_eating'], axis=1, inplace = True)
        self.data.drop(['behavior_personalHygine'], axis=1, inplace = True)
        self.data.drop(['intention_aggregation'], axis=1, inplace = True)
        self.data.drop(['intention_commitment'], axis=1, inplace = True)
        self.data.drop(['attitude_consistency'], axis=1, inplace = True)
        self.data.drop(['attitude_spontaneity'], axis=1, inplace = True)
        self.data.drop(['norm_significantPerson'], axis=1, inplace = True)
        self.data.drop(['norm_fulfillment'], axis=1, inplace = True)
        self.data.drop(['perception_vulnerability'], axis=1, inplace = True)
        self.data.drop(['perception_severity'], axis=1, inplace = True)
        self.data.drop(['motivation_strength'], axis=1, inplace = True)
    
        ###################
        # Normalizing dataset
        ###################
        x_1 = preprocessing.scale(self.data["motivation_willingness"])
        self.data["motivation_willingness"] = x_1
        x_2 = preprocessing.scale(self.data["socialSupport_emotionality"])
        self.data["socialSupport_emotionality"] = x_2
        x_3 = preprocessing.scale(self.data["socialSupport_appreciation"])
        self.data["socialSupport_appreciation"] = x_3
        x_4 = preprocessing.scale(self.data["socialSupport_instrumental"])
        self.data["socialSupport_instrumental"] = x_4
        x_5 = preprocessing.scale(self.data["empowerment_knowledge"])
        self.data["empowerment_knowledge"] = x_5
        x_6 = preprocessing.scale(self.data["empowerment_abilities"])
        self.data["empowerment_abilities"] = x_6
        x_7 = preprocessing.scale(self.data["empowerment_desires"])
        self.data["empowerment_desires"] = x_7
    
    def get_train_test_set(self):        
        ##################################
        # Spliting the dataset into training and test parts. 
        ##################################
        
        #Splitting datasets in training and test
        X = self.data[['socialSupport_emotionality','socialSupport_appreciation','socialSupport_instrumental','empowerment_knowledge','empowerment_abilities','empowerment_desires']]
        
        #creating the bias vector in dataset
        #bias = np.ones(len(X))
        #X.insert (0, 'bias', bias)
        
        Y = self.data[['ca_cervix']]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=101)

        return X_train, X_test, Y_train, Y_test

data = Preprocessing(url)
data.Exploratory_Data_Analysis()
X, X_test, y, Y_test = data.get_train_test_set()


# #(200, 2) (200,)
print(X.shape, y.shape)



input_neurons = 6
output_neurons = 2
samples = X.shape[0]
learning_rate = 0.001
lambda_reg = 0.01



def retreive(model_dict):
    W1 = model_dict['W1']
    b1 = model_dict['b1']
    W2 = model_dict['W2']
    b2 = model_dict['b2']
    return W1, b1, W2, b2


def backpropagation(x, y, model_dict, epochs):
    for i in range(epochs):
        W1, b1, W2, b2 = retreive(model_dict)
        z1, a1, probs = forward(x, model_dict)    # a1: (200,3), probs: (200,2)
        delta3 = np.copy(probs)
        delta3[range(x.shape[0]), y] -= 1      # (200,2)
        dW2 = (a1.T).dot(delta3)               # (3,2)
        db2 = np.sum(delta3, axis=0, keepdims=True)        # (1,2)
        delta2 = delta3.dot(W2.T) * (1 - np.power(np.tanh(z1), 2))
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # Add regularization terms
        dW2 += lambda_reg * np.sum(W2)  
        dW1 += lambda_reg * np.sum(W1)  
        # Update Weights: W = W + (-lr*gradient) = W - lr*gradient
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2
        # Update the model dictionary
        model_dict = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        # Print the loss every 50 epochs
        #if i%50 == 0:
            #print("Loss at epoch {} is: {:.3f}".format(i,loss(probs, y, model_dict)))
        error = loss(probs, y, model_dict)
    return model_dict, error




# Define Initial Weights
def init_network(input_dim, hidden_dim, output_dim):
    model = {}
    # Xavier Initialization 
    #[6,3]
    W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
    b1 = np.zeros((1, hidden_dim))
    #[3,2]
    W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
    b2 = np.zeros((1, output_dim))
    model['W1'] = W1
    model['b1'] = b1
    model['W2'] = W2
    model['b2'] = b2
    
    return model


def forward(x, model_dict):
    W1, b1, W2, b2 = retreive(model_dict)
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)  
    softmax = exp_scores / np.sum(exp_scores, axis = 1)
    s= softmax[[0, 1]].to_numpy()
    return z1, a1, s

def loss(softmax, y, model_dict):
    W1, b1, W2, b2 = retreive(model_dict)
    
    y_hat = []
    
    for i in softmax:
        if( i[0] >= i[1]):
            y_hat.append(0)
        else:
            y_hat.append(1)

    y_hat = np.reshape(y_hat, (y.shape[0], y.shape[1]))
    y_actual = np.array(y[['ca_cervix']])

    return(sum(y_actual != y_hat) / len(y))

def test(x, y, model_dict):
    #FORWARD
    W1, b1, W2, b2 = retreive(model_dict)
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    softmax = exp_scores / np.sum(exp_scores)
    s= softmax[[0, 1]].to_numpy()

    y_hat_test = []
    
    for i in s:
        if( i[0] >= i[1]):
            y_hat_test.append(0)
        else:
              y_hat_test.append(1)
              
    y_hat_test = np.reshape(y_hat_test, (y.shape[0], y.shape[1]))
    y_actual = np.array(y[['ca_cervix']])

    a = (sum(y_actual == y_hat_test) / len(y), y_hat_test)
    return a

# Now Let's start the action
model_dict = init_network(input_dim = input_neurons , hidden_dim = 3, output_dim = output_neurons)
model,train_error = backpropagation(X, y, model_dict, 50)
a, yh = test(X_test, Y_test, model_dict)


