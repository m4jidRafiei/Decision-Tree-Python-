# Original code from: https://github.com/m4jidRafiei/Decision-Tree-Python-
#
# Modified by a student to return the Digraph object instead of rendering it automatically.
# Modified to avoid error of mis-identification of graphviz nodes. Although I used a random
# generation and probabilistic cosmic rays might introduce equal IDs nevertheless.

from random import random
import math
from collections import deque
from graphviz import Digraph

class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.name = ""

# Simple class of Decision Tree
# Aimed for who want to learn Decision Tree, so it is not optimized
class DecisionTree(object):
    def __init__(self, sample, attributes, labels, criterion):
        self.sample = sample
        self.attributes = attributes
        self.labels = labels
        self.labelCodes = None
        self.labelCodesCount = None
        self.initLabelCodes()
        self.criterion = criterion
        # print(self.labelCodes)
        self.gini = None
        self.entropy = None
        self.root = None
        if(self.criterion == "gini"):
            self.gini = self.getGini([x for x in range(len(self.labels))])
        else:
            self.entropy = self.getEntropy([x for x in range(len(self.labels))])

    def initLabelCodes(self):
        self.labelCodes = []
        self.labelCodesCount = []
        for l in self.labels:
            if l not in self.labelCodes:
                self.labelCodes.append(l)
                self.labelCodesCount.append(0)
            self.labelCodesCount[self.labelCodes.index(l)] += 1

    def getLabelCodeId(self, sampleId):
        return self.labelCodes.index(self.labels[sampleId])

    def getAttributeValues(self, sampleIds, attributeId):
        vals = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in vals:
                vals.append(val)
        # print(vals)
        return vals

    def getEntropy(self, sampleIds):
        entropy = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1
        # print("-ge", labelCount)
        for lv in labelCount:
            # print(lv)
            if lv != 0:
                entropy += -lv/len(sampleIds) * math.log(lv/len(sampleIds), 2)
            else:
                entropy += 0
        return entropy

    def getGini(self, sampleIds):
        gini = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1
        # print("-ge", labelCount)
        for lv in labelCount:
            # print(lv)
            if lv != 0:
                gini += (lv/len(sampleIds)) ** 2
            else:
                gini += 0
        return 1 - gini

    def getDominantLabel(self, sampleIds):
        labelCodesCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCodesCount[self.labelCodes.index(self.labels[sid])] += 1
        return self.labelCodes[labelCodesCount.index(max(labelCodesCount))]

    def getInformationGain(self, sampleIds, attributeId):
        gain = self.getEntropy(sampleIds)
        attributeVals = []
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
        for vc, vids in zip(attributeValsCount, attributeValsIds):
            # print("-gig", vids)
            gain -= (vc/len(sampleIds)) * self.getEntropy(vids)
        return gain

    def getInformationGainGini(self, sampleIds, attributeId):
        gain = self.getGini(sampleIds)
        attributeVals = []
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
        for vc, vids in zip(attributeValsCount, attributeValsIds):
            # print("-gig", vids)
            gain -= (vc/len(sampleIds)) * self.getGini(vids)
        return gain

    def getAttributeMaxInformationGain(self, sampleIds, attributeIds):
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGain(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        try:
            maxvalue = attributesEntropy[maxId]
        except:
            maxvalue = 0
        return self.attributes[maxId], maxId, maxvalue

    def getAttributeMaxInformationGainGini(self, sampleIds, attributeIds):
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGainGini(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        try:
            maxvalue = attributesEntropy[maxId]
        except:
            maxvalue = 0
        return self.attributes[maxId], maxId, maxvalue

    def isSingleLabeled(self, sampleIds):
        label = self.labels[sampleIds[0]]
        for sid in sampleIds:
            if self.labels[sid] != label:
                return False
        return True

    def getLabel(self, sampleId):
        return self.labels[sampleId]

    def id3(self,gain_threshold, minimum_samples):
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        self.root = self.id3Recv(sampleIds, attributeIds, self.root, gain_threshold, minimum_samples)

    def id3Recv(self, sampleIds, attributeIds, root, gain_threshold, minimum_samples):
        root = Node() # Initialize current root
        if self.isSingleLabeled(sampleIds):
            root.value = self.labels[sampleIds[0]]
            return root
        # print(attributeIds)
        if len(attributeIds) == 0:
            root.value = self.getDominantLabel(sampleIds)
            return root
        if(self.criterion == "gini"):
            bestAttrName, bestAttrId, bestValue = self.getAttributeMaxInformationGainGini(sampleIds, attributeIds)
        else:
            bestAttrName, bestAttrId, bestValue = self.getAttributeMaxInformationGain(sampleIds, attributeIds)
        # print(bestAttrName)
        #if(bestValue > 0):
            #print("Best gain -> " + bestAttrName + "::" + str(bestValue) + "\n" )

        root.value = bestAttrName
        root.childs = []  # Create list of children

        if(bestValue < gain_threshold):
            Dominantlabel = self.getDominantLabel(sampleIds)
            root.value = Dominantlabel
            return root

        if(len(sampleIds) < minimum_samples):
            Dominantlabel = self.getDominantLabel(sampleIds)
            root.value = Dominantlabel
            return root

        for value in self.getAttributeValues(sampleIds, bestAttrId):
                # print(value)
                child = Node()
                child.value = value
                root.childs.append(child)  # Append new child node to current root
                childSampleIds = []
                for sid in sampleIds:
                    if self.sample[sid][bestAttrId] == value:
                        childSampleIds.append(sid)
                if len(childSampleIds) == 0:
                    child.next = self.getDominantLabel(sampleIds)
                else:
                    # print(bestAttrName, bestAttrId)
                    # print(attributeIds)
                    if len(attributeIds) > 0 and bestAttrId in attributeIds:
                        toRemove = attributeIds.index(bestAttrId)
                        attributeIds.pop(toRemove)

                    child.next = self.id3Recv(childSampleIds, attributeIds.copy(), child.next, gain_threshold, minimum_samples)
        return root

    def print_visualTree(self, render=True):
        dot = Digraph(comment='Decision Tree')
        if self.root:
            self.root.name = "root"
            roots = deque()
            roots.append(self.root)
            counter = 0
            while len(roots) > 0:
                root = roots.popleft()
#                 print(root.value)
                dot.node(root.name, root.value)
                if root.childs:
                    for child in root.childs:
                        counter += 1
#                         print('({})'.format(child.value))
                        child.name = str(random())
                        dot.node(child.name, child.value)
                        dot.edge(root.name,child.name)
                        if(child.next.childs):
                            child.next.name = str(random())
                            dot.node(child.next.name, child.next.value)
                            dot.edge(child.name,child.next.name)
                            roots.append(child.next)
                        else:
                            child.next.name = str(random())
                            dot.node(child.next.name, child.next.value)
                            dot.edge(child.name,child.next.name)

                elif root.next:
                    dot.node(root.next, root.next)
                    dot.edge(root.value,root.next)
#                     print(root.next)
#         print(dot.source)
        if render :
            try:
                dot.render('output/visualTree.gv', view=True)
            except:
                print("You either have not installed the 'dot' to visualize the decision tree or the reulted .pdf file is open!")
        return dot
    
##########################################################################################################
    
#from p_decision_tree.DecisionTree import DecisionTree

## Prepare the descriptive and target features -
## Obtaining lists with the names of the features is straightforward.
## For the values - we cast the column values to 'str' and extract them 

#ua_descriptive_features = ['SCHEDULED_DEPARTURE_CATEGORY', 'DISTANCE_CATEGORY', 'DAY_OF_WEEK']
#ua_descriptive_values = ua_flights[ua_descriptive_features].astype(str).values
#ua_target_values = ua_flights['DELAY'].astype(str).values

## Train an entropy-based Decision Tree for classifying delays
#ua_decision_tree_entropy = DecisionTree(sample=ua_descriptive_values,
#                                        attributes=ua_descriptive_features,
#                                        labels=ua_target_values,
#                                        criterion="entropy")

## Apply ID3 algorithm
#ua_decision_tree_entropy.id3(gain_threshold=0, minimum_samples=1000)

## Proceed to generate visual output with graphiz
## The resulting pdf file is submitted alongside the notebook.
#ua_decision_tree_entropy.print_visualTree(render=True)

## Print the total entropy of the used dataset
#print("Entropy-based Decision Tree: Entropy = %.5f" % ua_decision_tree_entropy.entropy)

##########################################################################################################

## shouldDraw() accepts a node and checks whether all children
## of its children nodes are identical. A set of those children 
## is then returned, allowing us to decide whether we should or
## should not draw the node in question.

## EXAMPLE - we SHOULD NOT draw the node 'DAY_OF_WEEK' when its parent is 'Night'.
## The reason is that, for every day of the week, the outcome is the same ('Acceptable_delay').
## For that reason, for each of the children (in this case - the days of the week) we check
## whether they themselves have children (then they are not parents of leaves) or if they do not
## (then they are parents of leaves). If they are parents of leaves, we can check whether the
## leaves are identical, and omit drawing unnecessary nodes.
 
def shouldDraw(node):
    # print(node.value)
    leaves = []
    if (node.childs):
        for child in node.childs:
            if (not(child.next.childs)):
                leaves.append(child.next.value)
    return set(leaves)

## printTree() is very much alike to print_visualTree(self, render=True) from the DecisionTree.py file:
## https://github.com/m4jidRafiei/Decision-Tree-Python-/blob/master/build/lib/p_decision_tree/DecisionTree.py
## CHANGES: 
##     - changed the name of deque() to 'nodes'
##     - after [if(child.next.childs)] (line 240 in original code) - added additional checks

def printTree(root, render=True):
    dot = Digraph(comment='Decision Tree')
    if root:
        root.name = "root"
        nodes = deque()
        nodes.append(root)
        while len(nodes) > 0:
            root = nodes.popleft()
            dot.node(root.name, root.value)
            if root.childs:
                for child in root.childs:
                    child.name = str(random())
                    dot.node(child.name, child.value)
                    dot.edge(root.name,child.name)
                    if(child.next.childs):
                        chld = child.next
                        decision_set = shouldDraw(chld)
                        # print(decision_set)
                        ## If all leaves are identical, skip creating nodes 
                        if (len(decision_set) == 1):
                            name = str(random())
                            dot.node(name, list(decision_set)[0])
                            dot.edge(child.name, name)
                        ## Otherwise, create the parent node and prepare children 
                        ## for further visualisation by enque()-ing them
                        else:        
                            child.next.name = str(random())
                            dot.node(child.next.name, child.next.value)
                            dot.edge(child.name,child.next.name)
            elif root.next:
                dot.node(root.next, root.next)
                dot.edge(root.value,root.next)
    if render:
        try:
            dot.render('output/visualTree.gv', view=True)
        except:
            print("You either have not installed the 'dot' to visualize the decision tree or the reulted .pdf file is open!")
    return dot

## Concrete example, based on the created decision tree
printTree(ua_decision_tree_entropy.root, render=True)
