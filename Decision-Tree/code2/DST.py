import itertools
import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import math

def gini_impurity(y):
    if isinstance(y, pd.Series):
        p = y.value_counts()/y.shape[0]
        gini = 1-np.sum(p**2)
        return(gini)

    else:
        raise('Object must be a Pandas Series.')

def entropy(y):
    if isinstance(y, pd.Series):
        a = y.value_counts()/y.shape[0]
        entropy = np.sum(-a*np.log2(a+1e-9))
        return(entropy)

    else:
        raise('Object must be a Pandas Series.')

def variance(y):
    if(len(y) == 1):
        return 0
    else:
        return y.var()

def information_gain(y, mask, func=entropy):
    a = sum(mask)
    b = mask.shape[0] - a
  
    if(a == 0 or b ==0): 
        ig = 0
  
    else:
        if y.dtypes != 'O':
            ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
        else:
            ig = func(y)-a/(a+b)*func(y[mask])-b/(a+b)*func(y[-mask])
  
    return ig

def categorical_options(a):
    a = a.unique()
    opciones = []
    for L in range(0, len(a)+1):
        for subset in itertools.combinations(a, L):
            subset = list(subset)
            opciones.append(subset)

    return opciones[1:-1]

def max_information_gain_split(x, y, func=entropy):
    split_value = []
    ig = [] 
    numeric_variable = True if x.dtypes != 'O' else False

  # Create options according to variable type
    if numeric_variable:
        options = x.sort_values().unique()[1:]
    else: 
        options = categorical_options(x)

  # Calculate ig for all values
    for val in options:
        mask =   x < val if numeric_variable else x.isin(val)
        val_ig = information_gain(y, mask, func)
        # Append results
        ig.append(val_ig)
        split_value.append(val)

  # Check if there are more than 1 results if not, return False
    if len(ig) == 0:
        return(None,None,None, False)

    else:
    #Get results with highest IG
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return(best_ig,best_split,numeric_variable, True)

def get_best_split(y, data):
    masks = data.drop(y, axis=1).apply(max_information_gain_split, y=data[y])

    if sum(masks.loc[3,:]) == 0:
        return(None, None, None, None)

    else:
        # Get only masks that can be splitted
        masks = masks.loc[:,masks.loc[3,:]]
        # Get the results for split with highest IG
        split_variable = masks.iloc[0].astype(np.float32).idxmax()
        #split_valid = masks[split_variable][]
        split_value = masks[split_variable][1] 
        split_ig = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)

def make_split(variable, value, data, is_numeric):
    if is_numeric:
        data_1 = data[data[variable] < value]
        data_2 = data[(data[variable] < value) == False]

    else:
        data_1 = data[data[variable].isin(value)]
        data_2 = data[(data[variable].isin(value)) == False]

    return(data_1,data_2)

def make_prediction(data, target_factor):
  # Make predictions
    if target_factor:
        pred = data.value_counts().idxmax()
    else:
        pred = data.mean()

    return pred

def train_tree(data,y, target_factor, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
      # Check that max_categories is fulfilled
    if counter==0:
        types = data.dtypes
        check_columns = types[types == "object"].index
        for column in check_columns:
            var_length = len(data[column].value_counts()) 
            if var_length > max_categories:
                raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

    # Check for depth conditions
    if max_depth == None:
        depth_cond = True

    else:
        if counter < max_depth:
            depth_cond = True

        else:
            depth_cond = False

    # Check for sample conditions
    if min_samples_split == None:
        sample_cond = True

    else:
        if data.shape[0] > min_samples_split:
            sample_cond = True
        else:
            sample_cond = False

  # Check for ig condition
    if depth_cond & sample_cond:
        var,val,ig,var_type = get_best_split(y, data)
        # If ig condition is fulfilled, make split 
        if ig is not None and ig >= min_information_gain:
            counter += 1
            left,right = make_split(var, val, data,var_type)

            # Instantiate sub-tree
            split_type = "<=" if var_type else "in"
            question =   "{} {}  {}".format(var,split_type,val)
            # question = "\n" + counter*" " + "|->" + var + " " + split_type + " " + str(val) 
            subtree = {question: []}

            # Find answers (recursion)
            yes_answer = train_tree(left,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)
            no_answer = train_tree(right,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

            if yes_answer == no_answer:
                subtree = yes_answer
            else:
                subtree[question].append(yes_answer)
                subtree[question].append(no_answer)

            # If it doesn't match IG condition, make prediction
        else:
            pred = make_prediction(data[y],target_factor)
            return pred

    # Drop dataset if doesn't match depth or sample conditions
    else:
        pred = make_prediction(data[y],target_factor)
        return pred

    return subtree

def clasificar_datos(observacion, arbol):
    question = list(arbol.keys())[0] 

    if question.split()[1] == '<=':
        if observacion[question.split()[0]] <= float(question.split()[2]):
            answer = arbol[question][0]
        else:
            answer = arbol[question][1]

    else:
        if observacion[question.split()[0]] in (question.split()[2]):
            answer = arbol[question][0]
        else:
            answer = arbol[question][1]

    # If the answer is not a dictionary
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return clasificar_datos(observacion, answer)

dataset = pd.read_csv("restaurant.csv")
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset["wait"], test_size=0.25, random_state=42)
# Train a decision tree using entropy
entropy_tree = train_tree(X_train, y_train, target_factor=True)
# Train a decision tree using gini index
gini_tree = train_tree(X_train, y_train, target_factor=True, criterion="gini")
# Evaluate the performance of the two trees on the test set
entropy_score = entropy_tree.score(X_test, y_test)
gini_score = gini_tree.score(X_test, y_test)

print("Entropy score:", entropy_score)
print("Gini score:", gini_score)