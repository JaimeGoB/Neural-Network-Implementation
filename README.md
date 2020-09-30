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





![Neural Network Flow](https://github.com/JaimeGoB/Neural-Network-Implementation/blob/master/images/Neural-Network-Flow.png)
![Neural Network Flow](https://github.com/JaimeGoB/Neural-Network-Implementation/blob/master/images/Neural-Network.png)
