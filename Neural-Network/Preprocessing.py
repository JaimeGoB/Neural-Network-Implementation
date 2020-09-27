import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
        sns.heatmap(data=correlation_matrix, annot=True).set_title('Heat Map')

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
        bias = np.ones(len(X))
        X.insert (0, 'bias', bias)
        
        Y = self.data[['ca_cervix']]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=101)

        return X_train, X_test, Y_train, Y_test