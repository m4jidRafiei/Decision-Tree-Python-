import math
from collections import deque
from graphviz import Digraph

class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None

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
         
                    child.next = self.id3Recv(childSampleIds, attributeIds, child.next, gain_threshold, minimum_samples)
        return root

    def print_visualTree(self):
        dot = Digraph(comment='Decision Tree')
        if self.root:
            roots = deque()
            roots.append(self.root)
            counter = 0
            while len(roots) > 0:
                root = roots.popleft()
#                 print(root.value)
                dot.node(root.value, root.value)
                if root.childs:
                    for child in root.childs:
                        counter += 1
#                         print('({})'.format(child.value))
                        dot.node(child.value, child.value)
                        dot.edge(root.value,child.value)
                        if(child.next.childs):
                            dot.node(child.next.value, child.next.value)
                            dot.edge(child.value,child.next.value)
                            roots.append(child.next)
                        else:
                            nodeName = ""
                            try: 
                                nodeName = child.next.value+str(counter)
                            except:
                                nodeName = ""+str(counter)
                            dot.node(nodeName, child.next.value)
                            dot.edge(child.value,nodeName)
                        
                elif root.next:
                    dot.node(root.next, root.next)
                    dot.edge(root.value,root.next)
#                     print(root.next)
#         print(dot.source) 
        try:
            dot.render('output/visualTree.gv', view=True)   
        except:
            print("Please close the resulted pdf file")
