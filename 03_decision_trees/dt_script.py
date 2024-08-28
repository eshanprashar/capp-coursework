import math
import sys
import pandas as pd
import csv


class Node:
    """
    Initializing node class to create tree
    A node may have children which are stored in a dictionary
    Has method add_child which adds attribute values and associated sub-trees as key-val pairs
    """

    def __init__(self, label, is_leaf=False, split_att=None):
        self.label = label
        self.split_att = split_att
        self.is_leaf = is_leaf
        self.children = {}

    def add_child(self, edge_label, child_node):
        self.children[edge_label] = child_node


def calculate_gini(data, attr):
    """
    Calculates the gini coefficient of a given dataset and an attribute
    """
    target_values = data[attr].unique()
    gini = 1
    total_size = len(data)
    for value in target_values:
        prob = len(data[data[attr] == value]) / total_size
        gini -= prob**2
    return gini


def calculate_gain(data, attr, target_col):
    """
    Calculates the information gain of splitting a dataset on a given attribute
    """
    original_gini = calculate_gini(data, target_col)
    split_gini = 0
    attr_values = data[attr].unique()
    total_size = len(data)
    for value in attr_values:
        subset = data[data[attr] == value]
        prob = len(subset) / total_size
        split_gini += prob * calculate_gini(subset, target_col)
    gain = original_gini - split_gini
    return gain


def calculate_split_info(data, attr):
    """
    Calculates the split information of splitting a dataset on a given attribute.
    """
    attr_values = data[attr].unique()
    total_size = len(data)
    split_info = 0
    for value in attr_values:
        subset = data[data[attr] == value]
        prob = len(subset) / total_size
        if len(subset) == 0:
            continue
        split_info -= prob * math.log(prob)
    return split_info


def calculate_gain_ratio(data, attr, target_col):
    """
    Calculates the gain ratio of splitting a dataset on a given attribute.
    """
    gain = calculate_gain(data, attr, target_col)
    split_info = calculate_split_info(data, attr)
    if split_info == 0:
        return 0
    return gain / split_info


def find_best_attribute(data, attributes, target_col):
    """
    Finds the best attribute based on gain ratio. \
    If two attributes have the same, return the one that occurs first
    """
    best_attr = ""
    best_ratio = 0.0
    for attr in attributes:
        ratio = calculate_gain_ratio(data, attr, target_col)
        if ratio > best_ratio:
            best_ratio = ratio
            best_attr = attr
        elif ratio == best_ratio and attr < best_attr:
            best_attr = attr
    return best_attr


def build_tree(data, attributes, target_col):
    """
    Builds a tree using training data, given its attributes and target column
    """
    # All base work for conditions
    target_counts = data[target_col].value_counts()
    max_count = target_counts.max()
    max_labels = target_counts[target_counts == max_count].index.tolist()
    best_attribute = find_best_attribute(data, attributes, target_col)

    if len(max_labels) == 1:
        label = max_labels[0]
    else:
        label = min(max_labels, key=lambda x: str(x))

    # Base case conditions
    if (
        len(target_counts) == 1
        or attributes is None
        or len(set(data[attributes].values.flatten())) == 1
        or best_attribute == ""
    ):
        return Node(label=label, split_att=None, is_leaf=True)

    # create a non-leaf node with the best attribute
    tree = Node(label=label, split_att=best_attribute, is_leaf=False)

    # for each possible value of the best attribute
    for value in set(data[best_attribute]):
        # recursively build subtree on subset of data with value of best attribute
        subset = data[data[best_attribute] == value].drop(columns=[best_attribute])
        subtree = build_tree(
            subset, [attr for attr in attributes if attr != best_attribute], target_col
        )
        # attach the subtree as a child of the non-leaf node
        tree.add_child(value, subtree)
    return tree


def classify(node, observation):
    """
    Sets a label for every row of testing data
    Inputs: Node object, row of test data
    Returns a label
    """
    if node.is_leaf:
        return str(node.label)
    else:
        attribute_value = observation.get(node.split_att)
        if attribute_value in node.children:
            child_node = node.children[attribute_value]
            return classify(child_node, observation)
        else:
            return str(node.label)


def go(training_filename, testing_filename):
    """
    Construct a decision tree using the training data and then apply
    it to the testing data.

    Inputs:
      training_filename (string): the name of the file with the
        training data
      testing_filename (string): the name of the file with the testing
        data

    Returns (list of strings or pandas series of strings): result of
      applying the decision tree to the testing data.
    """
    train = pd.read_csv(training_filename)
    with open(testing_filename, "r") as f:
        csv_reader = csv.DictReader(f)
        test_dict = list(csv_reader)

    target_col = train.columns[-1]
    attributes = list(train.columns[:-1])
    tree = build_tree(train, attributes, target_col)

    # Apply decision tree to testing data
    results = []
    for row in test_dict:
        classification = classify(tree, row)
        results.append(classification)
    print(results)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "usage: python3 {} <training filename> <testing filename>".format(
                sys.argv[0]
            )
        )
        sys.exit(1)

    for result in go(sys.argv[1], sys.argv[2]):
        print(result)