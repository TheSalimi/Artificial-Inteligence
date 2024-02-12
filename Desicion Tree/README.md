# Desicion Tree with Entropy and Gini index

## Introduction
### DT1.py
In the first step, implement the decision tree for the discrete data presented in the class slides. for testing, test the 12 data of the restaurant example.

![DT1](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/99872b4e-017e-436f-b775-273563c80e76)

![DT1-T](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/20f0d29e-ef95-4941-952a-da69c8e8760f)

### DT2.py
This Python script implements a decision tree classifier for predicting customer satisfaction based on various attributes. It utilizes information gain and Gini impurity as splitting criteria. The script reads data from a CSV file, discretizes numerical attributes, and builds decision trees using both entropy and Gini impurity. Accuracy is evaluated using a randomly selected subset of the dataset.

#### output : 
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/3da20e90-11df-4a43-bb9e-73626bf1f69f)
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/dfa19342-b544-4567-b4e8-65ba3d7b116f)


## Gini index VS Entropy
The Gini index and entropy are both metrics used in decision tree algorithms to measure the impurity of a dataset and guide the tree-building process. Here are the main differences between the two:

#### output : 
![image](https://github.com/TheSalimi/Artificial-Inteligence/assets/108394058/3af0e360-7856-4a10-9a46-69c1111e1edd)

### Formula
Gini Index: It measures the probability of a random sample being misclassified. It ranges from 0 to 1, where 0 indicates perfect purity (all samples belong to one class) and 1 indicates maximum impurity (equal distribution among all classes).

Entropy: It measures the uncertainty or randomness in a dataset. It ranges from 0 to 1, where 0 indicates perfect purity (all samples belong to one class) and 1 indicates maximum impurity (equal probability of all classes).

### Calculation
Gini Index: It calculates impurity by summing the squared probabilities of each class.

Entropy: It calculates impurity using the information-theoretic concept of entropy, which considers the distribution of classes in the dataset.

### Sensitivity to Class Imbalance:
Gini Index: It tends to be less sensitive to class imbalance compared to entropy.

Entropy: It can be sensitive to class imbalance because it focuses on the distribution of classes.

### Decision Boundary:
Gini Index: It tends to produce binary splits in the decision tree.

Entropy: It may create more balanced splits because it considers the distribution of classes.

###### Which one is better depends on the specific dataset and the problem at hand:

Gini Index is often preferred for its efficiency and simplicity, especially when dealing with binary classification problems or datasets with imbalanced class distributions.
Entropy is preferred in some cases, particularly when the dataset is well balanced across classes and there's a desire for more balanced splits in the decision tree.
