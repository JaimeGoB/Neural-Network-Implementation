# Section 0 - How to run the code
 #### Structure of Project
 + Neural-Network-Implementation 
 Has the source code.
 
 + Report 
 Contains the report for our Neural Network.
 
 #### Libraries Pre-Requisites:

 + pandas, numpy, matplotlib.pyplot, seaborn

 + #### The files cannot be run through command line.

 + #### They have to be run in Spyer IDE from Anaconda.

 ### STEPS:

 1) Open 'Neural-Network' folder using **Sypder IDE from Anaconda**
 **Maker sure the following files are inside of the folder.**
 - Activation_Function.py
 - driver.py
 - Neural_Network_Builder.py
 - Neural_Network.py
 - Optimizer.py
 - Preprocessing.py

 2) Read the ‘accuracy_for_different_parameters.csv’ to see proof of the use of different parameters then **Delete ‘accuracy_for_different_parameters.csv’.**

 3) Once you have open the 'Neural-Network' folder in Sypder and have deleted all .txt files here. We can proceed to run the code.

 **Only open driver.py. DO NOT OPEN ANY OTHER FILE. DRIVER.PY CONTROLS ALL FILES. PLEASE SEE CLASS DIAGRAM BELOW**

 - Option 1: Click play button on Sypder IDE. This button will run the entire file.

 - Option 2: Select Lines 1-60 from driver.py and hit run.

 A ‘accuracy_for_different_parameters.csv’ file be created at the end of running driver.py

 **It is the the output of the neural network in a tabular format using different parameters.**

# Neural-Network-Implementation

## The Dataset:
Unfortunately, cervical cancer (Ca Cervix) is a serious problem for women all over the world.
However, I can be preventable if detected early. Early prevention is a challenging task, but some studies
have shown that behavioral patterns might help to early detect in this horrible disease.
Using a neural network, we will classify if a person is likely to have cervical cancer based on certain
behavioral patterns.
 
#### Attribute Information of the Dataset:
This dataset consist of 18 attribute (comes from 8 variables, the name of variables is the first word in each attribute)
1) behavior_eating
2) behavior_personalHygine
3) intention_aggregation
4) intention_commitment
5) attitude_consistency
6) attitude_spontaneity
7) norm_significantPerson
8) norm_fulfillment
9) perception_vulnerability
10) perception_severity 
11) motivation_strength
12) motivation_willingness 
13)socialSupport_emotionality
14)socialSupport_appreciation
15)socialSupport_instrumental 
16) empowerment_knowledge 
17) empowerment_abilities
18) empowerment_desires
19) ca_cervix (this is class attribute, 1=has cervical cancer, 0=no cervical cancer)

After performing exploratory data analysis on the dataset and other methods. We concluded
that we the best attributes to fit a Neural Network classifier are:

- motivation_willingness 
- socialSupport_emotionality 
- socialSupport_appreciation
- socialSupport_instrumental 
- empowerment_knowledge
- empowerment_abilities
- empowerment_desires



## Neural Network Architecture
Our Neural Network is a shallow network of hidden layer.

N is the number of observations.

X is the feature vector from the best attributes we picked to fit a classifier (Input Layer). Size [N x 7] H is the for the hidden layer. hij is a single neuron in H. Size [N x 7]

W1ij are the weights 1 between input layer and hidden layer.

W2ij are the weights 2 between output layer and hidden layer.

y which is a vector of size [N x 1] containing actual values.

y_hat is a vector of size [N x 1] containing predicted values for target(y).

![Neural Network Flow](https://github.com/JaimeGoB/Neural-Network-Implementation/blob/master/images/nn-architecture.png)


## Neural Network UML and Class Diagram:
The process of our algorithm is as follows:
We preprocess the dataset and create the Neural Network using the training set. We then test our model with testing set to get results and then output them in a tabular format (.csv file).
![Neural Network Flow](https://github.com/JaimeGoB/Neural-Network-Implementation/blob/master/images/Neural-Network-Flow.png)

The class diagram below gives a more detailed explanation of how the Neural Network is created, train and test.
![Neural Network Flow](https://github.com/JaimeGoB/Neural-Network-Implementation/blob/master/images/Neural-Network.png)

## Neural Network Results
The following table is the most recent iteration of the neural net using the tanh function as the activation function:
![Neural Network Flow](https://github.com/JaimeGoB/Neural-Network-Implementation/blob/master/images/accuracy.png)
