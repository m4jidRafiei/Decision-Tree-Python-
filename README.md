# Visual Decision Tree Based on Categorical Attributes 
-------------------

As you may know "scikit-learn" library in python is not able to make a decision tree based on categorical data, and you have to convert categorical data to numerical before passing them to the classifier method. Also, the resulted decision tree is a binary tree while a decision tree does not need to be binary.

Here we provide a library which is able to make a visual decision tree based on categorical data. You can read more about decision trees [here](https://en.wikipedia.org/wiki/Decision_tree).

## Features
--------------------

The main algorithm which is used is ID3 with the following features:

* Information gain based on [entropy](https://en.wikipedia.org/wiki/Decision_tree_learning)
* Information gain based on [gini](https://en.wikipedia.org/wiki/Decision_tree_learning)
* Some pruning capabilities like:
	* Minimum number of samples
	* Minimum information gain
* The resulted tree is not binary

## Requirements
--------------------

You can find all the requirements in "requirements.txt" file, and it can be installed easily by the following command:

* pip install -r requirements.txt 

Also to be able to see visual tree, you need to install graphviz package. [Here](https://www.graphviz.org/download/) you can find the right package with respect to your operation system. 


## Usage
--------------------

```python

from DecisionTree import DecisionTree
import pandas as pd

#Reading CSV file as data set by Pandas
data = pd.read_csv('playtennis.csv')
columns = data.columns

#All columns except the last one are descriptive by default
descriptive_features = columns[:-1]
#The last column is considered as label
label = columns[-1]

#Converting all the columns to string 
for column in columns:
    data[column]= data[column].astype(str)
   
data_descriptive = data[descriptive_features].values
data_label = data[label].values

#Calling DecisionTree constructor (the last parameter is criterion which can also be "gini")
decisionTree = DecisionTree(data_descriptive.tolist(), descriptive_features.tolist(), data_label.tolist(), "entropy")

#Here you can pass pruning features (gain_threshold and minimum_samples)
decisionTree.id3(0,0)

#Visualizing decision tree by Graphviz
decisionTree.print_visualTree()

print("System entropy: ", format(decisionTree.entropy))
print("System gini: ", format(decisionTree.gini))


``` 

