import pandas as pd
import math
import numpy as np

# Load dataset
data = pd.read_csv(r"C:\Users\Shubham\Desktop\dicision_tree\datasett.csv")
features = [feat for feat in data.columns if feat != "answer"]

# Node class for the tree
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.pred = ""

# Entropy calculation
def entropy(examples):
    pos = sum(examples["answer"] == "yes")
    neg = sum(examples["answer"] == "no")
    if pos == 0 or neg == 0:
        return 0.0
    p = pos / (pos + neg)
    n = neg / (pos + neg)
    return -(p * math.log(p, 2) + n * math.log(n, 2))

# Information Gain
def info_gain(examples, attr):
    uniq = np.unique(examples[attr])
    gain = entropy(examples)
    for u in uniq:
        subdata = examples[examples[attr] == u]
        sub_e = entropy(subdata)
        gain -= (len(subdata) / len(examples)) * sub_e
    return gain

# ID3 Algorithm
def ID3(examples, attrs):
    root = Node()

    # If all examples have the same answer or no attributes left
    if entropy(examples) == 0 or not attrs:
        root.isLeaf = True
        root.pred = examples["answer"].mode()[0]
        return root

    max_gain = -1
    max_feat = None
    for feature in attrs:
        gain = info_gain(examples, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature

    if max_feat is None:
        root.isLeaf = True
        root.pred = examples["answer"].mode()[0]
        return root

    root.value = max_feat
    uniq = np.unique(examples[max_feat])
    for u in uniq:
        subdata = examples[examples[max_feat] == u]
        if subdata.empty:
            leaf = Node()
            leaf.isLeaf = True
            leaf.value = u
            leaf.pred = examples["answer"].mode()[0]
            root.children.append(leaf)
        elif entropy(subdata) == 0.0:
            leaf = Node()
            leaf.isLeaf = True
            leaf.value = u
            leaf.pred = subdata["answer"].values[0]
            root.children.append(leaf)
        else:
            child = Node()
            child.value = u
            new_attrs = [a for a in attrs if a != max_feat]
            subtree = ID3(subdata, new_attrs)
            child.children.append(subtree)
            root.children.append(child)
    return root

# Print the tree
def printTree(root, depth=0):
    print("\t" * depth + root.value, end="")
    if root.isLeaf:
        print(" ->", root.pred)
    else:
        print()
        for child in root.children:
            print("\t" * (depth + 1) + "[" + child.value + "]")
            if child.children:
                printTree(child.children[0], depth + 2)

# Classify a new sample
def classify(root, sample):
    for child in root.children:
        if child.value == sample[root.value]:
            if child.isLeaf:
                print(f"Predicted label for {sample} is: {child.pred}")
                return
            else:
                classify(child.children[0], sample)
                return
    print(f"No matching path found for {sample}")

# Train the model
root = ID3(data, features)
print("Decision Tree:")
printTree(root)
print("------------------")

# Test sample
new = {"outlook": "sunny", "temperature": "hot", "humidity": "normal", "wind": "strong"}
classify(root, new)
