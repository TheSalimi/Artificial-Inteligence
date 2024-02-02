import json
import math
import numpy as np
inputValues = {}

data = np.array([
    ( {'Alt': 1 , 'Bar': 0, 'Fri': 0, 'Hun': 1, 'Pat':'Full', 'Price': 1, 'Rain': 0, 'Res': 0, 'Type':'Thai', 'Est':30}  , 0  ),
    ( {'Alt': 0 , 'Bar': 1, 'Fri': 0, 'Hun': 0, 'Pat':'Some', 'Price': 1, 'Rain': 0, 'Res': 0, 'Type':'Burger', 'Est':0}  , 1 ),
    ( {'Alt': 1 , 'Bar': 0, 'Fri': 0, 'Hun': 1, 'Pat':'Some', 'Price': 3, 'Rain': 0, 'Res': 1, 'Type':'French', 'Est':0}  , 1 ),
    ( {'Alt': 1 , 'Bar': 0, 'Fri': 1, 'Hun': 1, 'Pat':'Full', 'Price': 1, 'Rain': 1, 'Res': 0, 'Type':'Thai', 'Est':10}  , 1  ),
    ( {'Alt': 1 , 'Bar': 0, 'Fri': 1, 'Hun': 0, 'Pat':'Full', 'Price': 3, 'Rain': 0, 'Res': 1, 'Type':'French', 'Est':60}  , 0),
    ( {'Alt': 0 , 'Bar': 1, 'Fri': 0, 'Hun': 1, 'Pat':'Some', 'Price': 2, 'Rain': 1, 'Res': 1, 'Type':'Italian', 'Est':0}  , 1),
    ( {'Alt': 0 , 'Bar': 1, 'Fri': 0, 'Hun': 0, 'Pat':'None', 'Price': 1, 'Rain': 1, 'Res': 0, 'Type':'Burger', 'Est':0}  , 0 ),
    ( {'Alt': 0 , 'Bar': 0, 'Fri': 0, 'Hun': 1, 'Pat':'Some', 'Price': 2, 'Rain': 1, 'Res': 1, 'Type':'Thai', 'Est':0}  , 1   ),
    ( {'Alt': 0 , 'Bar': 1, 'Fri': 1, 'Hun': 0, 'Pat':'Full', 'Price': 1, 'Rain': 1, 'Res': 0, 'Type':'Burger', 'Est':60}  , 0),
    ( {'Alt': 1 , 'Bar': 1, 'Fri': 1, 'Hun': 1, 'Pat':'Full', 'Price': 3, 'Rain': 0, 'Res': 1, 'Type':'Italian', 'Est':10}  ,0),
    ( {'Alt': 0 , 'Bar': 0, 'Fri': 0, 'Hun': 0, 'Pat':'None', 'Price': 1, 'Rain': 0, 'Res': 0, 'Type':'Thai', 'Est':0}  , 0   ),
    ( {'Alt': 1 , 'Bar': 1, 'Fri': 1, 'Hun': 1, 'Pat':'Full', 'Price': 1, 'Rain': 0, 'Res': 0, 'Type':'Burger', 'Est':30}  , 1),
])

def makeKeyValues(data):
    for input, _ in data:
        for attr, value in input.items():
            if attr not in inputValues:
                inputValues[attr] = set()
            inputValues[attr].add(value)
            
def entropy(data):
    if len(data) == 0:
        return 0
    else:
        positive = sum(1 for _, label in data if label == 1)
        negative = len(data) - positive

        if positive == 0 or negative == 0:
            return 0
        else:
            positivePossibility = positive / len(data)
            negativePossibility = negative / len(data)
            return -positivePossibility * math.log2(positivePossibility) - negativePossibility * math.log2(negativePossibility)

def giniIndex(data):
    if len(data) == 0:
        return 0
    else:
        positiveNumber = sum(1 for _, label in data if label == 1)
        negativeNumber = len(data) - positiveNumber

        p_negative = negativeNumber / len(data)
        p_positive = positiveNumber / len(data)

        giniIndex = 1 - (p_positive ** 2 + p_negative ** 2)
        return giniIndex

def entropyInformaionGain(data, attribute):
    entropy_before = entropy(data)
    num_data = len(data)
    weighted_entropy_after = 0.0

    for value in set(inputValues[attribute]):
        subset = [(x, y) for x, y in data if x[attribute] == value]
        weight = len(subset) / num_data
        weighted_entropy_after += weight * entropy(subset)

    return entropy_before - weighted_entropy_after

def giniImpurity(data, attribute):
    weightedGini = 0.0

    for value in set(inputValues[attribute]):
        subset = [(x, y) for x, y in data if x[attribute] == value]
        weight = len(subset) / len(data)
        weightedGini += weight * giniIndex(subset)

    return weightedGini

def giniInformaionGain(data, attribute):
    gini_before = giniIndex(data)
    gini_impurity_before = giniImpurity(data, attribute)
    return gini_before - gini_impurity_before

def chooseBestAttr(data, attributes, criterion):
    if criterion == 'Entropy':
        best_criterion_function = entropyInformaionGain
    elif criterion == 'Gini':
        best_criterion_function = giniInformaionGain

    best_info_gain = 0
    best_attribute = None

    for attribute in attributes:
        info_gain = best_criterion_function(data, attribute)
        if info_gain > best_info_gain:
            best_attribute = attribute
            best_info_gain = info_gain

    return best_attribute

def buildTree(data, attributes, criterion):
    labels = [label for _, label in data]

    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not attributes:
        return max(set(labels), key=labels.count)

    bestAttr = chooseBestAttr(data, attributes, criterion)

    tree = {bestAttr: {}}
    attributes.remove(bestAttr)

    for value in set(inputValues[bestAttr]):
        subset = [(x, y) for x, y in data if x[bestAttr] == value]

        if not subset:
            tree[bestAttr][value] = max(set(labels), key=labels.count)
        else:
            tree[bestAttr][value] = buildTree(subset, attributes.copy(), criterion)

    return tree

makeKeyValues(data)
names = list(inputValues.keys()).copy()

eTree = buildTree(data, names.copy(), 'Entropy')
print("Entropy:")
print(json.dumps(eTree, indent=2))

gTree = buildTree(data, names.copy(), 'Gini')
print("Gini:")
print(json.dumps(gTree, indent=2))



