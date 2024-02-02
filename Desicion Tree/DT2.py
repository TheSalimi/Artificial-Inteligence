import math
import csv
import random
import numpy as np
import math;
abs = 'airplane.csv'

attr_values = {}

def predict(tree, example):
    if isinstance(tree, dict):
        attribute = list(tree.keys())[0]
        value = example.get(attribute)

        if value in tree[attribute]:
            subtree = tree[attribute][value]
            return predict(subtree, example)
        else:
            return None
    else:
        return tree
      
def makeInputValues(data):
    for example, _ in data:
        for attr, value in example.items():
            if attr not in attr_values:
                attr_values[attr] = set()
            attr_values[attr].add(value)

def entropy(data):
    num_data = len(data)
    if num_data == 0:
        return 0

    num_positives = sum(1 for _, label in data if label == 'satisfied')
    num_negatives = num_data - num_positives

    if num_positives == 0 or num_negatives == 0:
        return 0

    p_positive = num_positives / num_data
    p_negative = num_negatives / num_data
    return -p_positive * math.log2(p_positive) - p_negative * math.log2(p_negative)

def gini_index(data):
    num_data = len(data)
    if num_data == 0:
        return 0

    num_positives = sum(1 for _, label in data if label == 'satisfied')
    num_negatives = num_data - num_positives

    p_positive = num_positives / num_data
    p_negative = num_negatives / num_data

    gini = 1 - (p_positive ** 2 + p_negative ** 2)
    return gini

def information_gain_entropy(data, attribute):
    entropy_before = entropy(data)
    num_data = len(data)
    weighted_entropy_after = 0.0

    for value in set(attr_values[attribute]):
        subset = [(x, y) for x, y in data if x[attribute] == value]
        weight = len(subset) / num_data
        weighted_entropy_after += weight * entropy(subset)

    return entropy_before - weighted_entropy_after

def gini_impurity(data, attribute):
    num_data = len(data)
    weighted_gini_after = 0.0

    for value in set(attr_values[attribute]):
        subset = [(x, y) for x, y in data if x[attribute] == value]
        weight = len(subset) / num_data
        weighted_gini_after += weight * gini_index(subset)

    return weighted_gini_after

def information_gain_gini(data, attribute):
    gini_before = gini_index(data)
    gini_impurity_before = gini_impurity(data, attribute)
    return gini_before - gini_impurity_before

def select_best_attribute(data, inputs, criterion):
    if criterion == 'Gini':
        best_criterion_function = information_gain_gini
    elif criterion == 'Entropy':
        best_criterion_function = information_gain_entropy
    else:
        raise ValueError("Invalid criterion")

    best_info_gain = -1
    best_attribute = None

    for attribute in inputs:
        info_gain = best_criterion_function(data, attribute)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute = attribute

    return best_attribute

def CalcAccuracy(giniDecisionTree,entropyDecisionTree):
    with open(abs, 'r') as csvfile:
        r = csv.DictReader(csvfile)
        c =0
        giniPredictCnt =0
        entropyPredictCnt =0

        for row in r :
            if c == 2000 :
                break
            c+=1

            dic = {key: value for key, value in row.items()}
            valuesatisfication = dic.pop('satisfaction')
            dic.pop('')
            dic.pop('id')

            gRes = predict(giniDecisionTree,dic)
            if gRes != None and str(gRes) == valuesatisfication:
                giniPredictCnt += 1

            eRes = predict(entropyDecisionTree,dic)
            if eRes != None and str(eRes) == valuesatisfication:
                entropyPredictCnt += 1

    return ((float(giniPredictCnt)/2000)*100, (float(entropyPredictCnt)/2000)*100)

def BuildTree(data, inputs, criterion):
    labels = [label for _, label in data]

    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if not inputs:
        return max(set(labels), key=labels.count)

    best_attribute = select_best_attribute(data, inputs, criterion)

    tree = {best_attribute: {}}
    inputs.remove(best_attribute)

    for value in set(attr_values[best_attribute]):
        subset = [(x, y) for x, y in data if x[best_attribute] == value]

        if not subset:
            tree[best_attribute][value] = max(set(labels), key=labels.count)
        else:
            tree[best_attribute][value] = BuildTree(subset, inputs.copy(), criterion)

    return tree

def Discreting(length, value , min, max):
    return math.floor((float(value) - min) / ((max - min) / length))

def ReadFromCSV(filename):
    data = np.genfromtxt(abs, delimiter=',', skip_header=True)
    columnData = data[:, 4]

    minAge = np.min(columnData)
    maxAge = np.max(columnData)

    columnData = data[:, 7]

    minFlightDistance = np.min(columnData)
    maxFlightDistance = np.max(columnData)

    columnData = data[:, 22]

    minDdelay = np.min(columnData)
    maxDdelay  = np.max(columnData)

    
    with open(filename, 'r') as csvfile:
        r = csv.DictReader(csvfile)
        satisfiedList = []
        notsatisfiedList = []

        for _ in range(2000):
            next(r, None)
        
        for row in r :            
            example = {key: value for key, value in row.items()}
            if example['satisfaction'] == 'satisfied' :
                satisfiedList.append(example)
            else:
                notsatisfiedList.append(example)

        random.shuffle(satisfiedList)
        random.shuffle(notsatisfiedList)
        selectedList = satisfiedList[:11000] + notsatisfiedList[:11000]
            
    data = []

    for row in selectedList:
        label = row.pop('satisfaction')
        row.pop('id')
        row['Age'] = Discreting(4,row['Age'],minAge,maxAge)
        row['Flight Distance'] = Discreting(8,row['Flight Distance'],minFlightDistance,maxFlightDistance)
        row['Departure Delay in Minutes'] = Discreting(8,row['Departure Delay in Minutes'],minDdelay,maxDdelay)
        row.pop('')
        data.append((row, label))

    return data



data = ReadFromCSV(abs)
makeInputValues(ReadFromCSV(abs))
inputs = list(attr_values.keys())

giniDecisionTree = BuildTree(data, inputs.copy(), 'Gini')
entropyDecisionTree = BuildTree(data, inputs.copy(), 'Entropy')

print('gini tree : ')
print(giniDecisionTree)

print('entropy tree')
print(entropyDecisionTree)

with open(abs, 'r') as csvFile:
    r = csv.DictReader(csvFile)
    cnt = 0
    giniPredictCount = 0
    entropyPredictCount = 0
    for row in r :
        if cnt == 2000 :
            break

        dic = {key: value for key, value in row.items()}
        dic.pop('')
        dic.pop('id')
        dic.pop('satisfaction')
        
        if predict(entropyDecisionTree,dic) != None :
            entropyPredictCount += 1

        if predict(giniDecisionTree,dic) != None :
            giniPredictCount += 1
        
        cnt+=1
        
precision = CalcAccuracy(giniDecisionTree, entropyDecisionTree)
print('Entropy accuracy : ', precision[1] )
print('Gini accuracy : ', precision[0] ) 