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

   
