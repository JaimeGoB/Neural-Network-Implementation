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

y which is a vector of size [57 x 1] containing actual values.

y_hat is a vector of size [57 x 1] containing predicted values for target(y).

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
