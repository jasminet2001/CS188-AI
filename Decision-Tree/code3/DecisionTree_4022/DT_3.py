import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

train = pd.read_csv('onlinefraud.csv')

transc_type = {
    "PAYMENT": 1,
    "TRANSFER": 2,
    "CASH_OUT": 3,
    "CASH_IN": 4,
    "DEBIT": 5
}
train["type"] = train["type"].apply(lambda x: transc_type[x])
train["nameOrig"] = train["nameOrig"].apply(
    lambda x: x.replace('C', '') if isinstance(x, str) else x)
train["nameDest"] = train["nameDest"].apply(
    lambda x: x.replace('M', '') if isinstance(x, str) else x)
train["nameDest"] = train["nameDest"].apply(
    lambda x: x.replace('C', '') if isinstance(x, str) else x)

train.loc[train['amount'] <= 1000, 'amount'] = 0
train.loc[(train['amount'] > 1000) & (
    train['amount'] <= 10000), 'amount'] = 1
train.loc[(train['amount'] > 10000) & (
    train['amount'] <= 50000), 'amount'] = 2
train.loc[train['amount'] > 50000, 'amount'] = 3


class Node:
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute
        self.value = value
        self.label = label
        self.children = {}

    def add_child(self, value, child_node):
        self.children[value] = child_node


def PLURALITY_VALUE(examples):
    return np.argmax(np.bincount(examples.iloc[:, -1]))


def find_attribute_index(attribute, attributes):
    return attributes.index(attribute)


def select_best_attribute_entropy(attributes, examples):
    best_attribute = None
    best_gain = -1
    for attribute in attributes:
        entropy = entropy_calc(examples[attribute])
        if best_attribute is None or entropy < best_gain:
            best_attribute = attribute
            best_gain = entropy
    return best_attribute


def entropy_calc(examples):
    unique_values, counts = np.unique(examples, return_counts=True)
    probabilities = counts / len(examples)
    return -np.sum(probabilities * np.log2(probabilities))


def LEARN_DECISION_TREE(examples, attributes, target, criterion='entropy'):
    if len(np.unique(target)) == 1:
        return Node(label=target.iloc[0])
    elif len(attributes) == 0:
        return Node(label=PLURALITY_VALUE(examples))
    else:
        if criterion == 'entropy':
            best_attribute = select_best_attribute_entropy(
                attributes, examples)
        else:
            best_attribute = select_best_attribute_gini(attributes, examples)
        tree = Node(attribute=best_attribute)
        attribute_index = find_attribute_index(best_attribute, attributes)
        unique_values = np.unique(examples[best_attribute])
        for value in unique_values:
            subset = examples[examples[best_attribute] == value]
            if len(subset) == 0:
                tree.add_child(value, Node(label=PLURALITY_VALUE(examples)))
            else:
                new_attributes = attributes[:attribute_index] + \
                    attributes[attribute_index + 1:]
                subtree = LEARN_DECISION_TREE(
                    subset, new_attributes, subset.iloc[:, -1], criterion)
                tree.add_child(value, subtree)
        return tree


def select_best_attribute_gini(attributes, examples):
    best_attribute = None
    best_gain = -1
    for attribute in attributes:
        gini = gini_calc(examples[attribute])
        if best_attribute is None or gini < best_gain:
            best_attribute = attribute
            best_gain = gini
    return best_attribute


def gini_calc(examples):
    unique_values, counts = np.unique(examples, return_counts=True)
    probabilities = counts / len(examples)
    return 1 - np.sum(probabilities ** 2)


def print_tree(node, indent="", file=None):
    if node.label is not None:
        output = indent + "Label: " + str(node.label)
        if file:
            print(output, file=file)
        else:
            print(output)
    else:
        output = indent + "Attribute: " + str(node.attribute)
        if file:
            print(output, file=file)
        else:
            print(output)
        for value, child_node in node.children.items():
            output = indent + "Value: " + str(value)
            if file:
                print(output, file=file)
            else:
                print(output)
            print_tree(child_node, indent + "  ", file)


def predict(node, instance, attributes):
    if node.label is not None:
        return node.label
    attribute_value = instance[attributes.index(node.attribute)]
    if attribute_value in node.children:
        child_node = node.children[attribute_value]
        return predict(child_node, instance, attributes)
    else:
        return None


def test_decision_tree(tree, test_data, attributes):
    predictions = []
    for instance in test_data:
        prediction = predict(tree, instance, attributes)
        predictions.append(prediction)
    return predictions


def calculate_accuracy(predictions, actual_labels):
    correct_predictions = 0
    total_predictions = len(predictions)

    for i in range(total_predictions):
        if predictions[i] == actual_labels[i]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def random_rows_after_2000(data_frame, num_rows):
    total_rows = data_frame.shape[0]
    eligible_indices = list(range(2000, total_rows))

    selected_indices = random.sample(eligible_indices, num_rows)
    selected_rows = data_frame.iloc[selected_indices]

    return selected_rows


train_data = random_rows_after_2000(train, 8000)
target = train_data.iloc[:, -1]
attributes = list(train.columns)[:-1]

# Train and test the decision tree with entropy
DT_entropy = LEARN_DECISION_TREE(
    train_data, attributes, target, criterion='entropy')

print("Decision Tree (Entropy):")
print_tree(DT_entropy)

NumberOfTestData = 2000
test_data_results = np.array(train.head(NumberOfTestData))[:, -1]
test_data = np.array(train.head(NumberOfTestData))[:, :-1]
predictions_entropy = test_decision_tree(DT_entropy, test_data, attributes)
accuracy_entropy = calculate_accuracy(predictions_entropy, test_data_results)
print("Accuracy (Entropy):", accuracy_entropy * 100)

# Train and test the decision tree with Gini index
DT_gini = LEARN_DECISION_TREE(train_data, attributes, target, criterion='gini')

print("Decision Tree (Gini):")
print_tree(DT_gini)

predictions_gini = test_decision_tree(DT_gini, test_data, attributes)
accuracy_gini = calculate_accuracy(predictions_gini, test_data_results)
print("Accuracy (Gini):", accuracy_gini * 100)

# Plot the results
criteria = ['Entropy', 'Gini']
accuracies = [accuracy_entropy * 100, accuracy_gini * 100]

plt.figure(figsize=(10, 5))
plt.bar(criteria, accuracies, color=['blue', 'green'])
plt.xlabel('Criteria')
plt.ylabel('Accuracy (%)')
plt.title('Decision Tree Accuracy Comparison: Entropy vs Gini')
plt.show()
