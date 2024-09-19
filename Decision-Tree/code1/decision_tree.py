import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import math

class TreeNode:
    def __init__(self, feature=None, amountThr=None,g=None, Entropy=None, Left=None, Right=None,leafVal=None,gini=None):
        self.feature = feature
        self.amountThr = amountThr
        self.Left = Left
        self.Right = Right
        self.leafVal = leafVal
        self.gain = g
        self.Entropy = Entropy
        self.gini = gini
    def IsLeaf(self):
        return self.leafVal is not None

class DecisionTree:
    def __init__(self, MaxDepth=50, featureCount=None, criterion='entropy'):
        self.MaxDepth=MaxDepth
        self.featureCount=featureCount
        self.root=None
        self.criterion = criterion

    def fitRoot(self, X, y):
        if self.featureCount == None:
            self.featureCount = X.shape[1]
        else:
            self.featureCount = min(X.shape[1],self.featureCount)
        if self.criterion == 'gini':
            self.root = self.ExpandTreeGini(X, y)
        else:
            self.root = self.ExpandTree(X, y)

    def ExpandTreeGini(self, X, y, depth=0):
        sampleCount = X.shape[0]
        featureCount = X.shape[1]
        LabelCount = len(np.unique(y))
        
        if depth >= self.MaxDepth or LabelCount == 1 or sampleCount < 2:
            c = Counter(y)
            # print("counter", c)
            leaf_Val = c.most_common(1)[0][0]
            return TreeNode(leafVal=leaf_Val)

        best_feature, best_thresh, Bestgain, best_gini = self.SplitWithBestGini(X, y, featureCount)
        leftIdxs, rightIdxs = self.Splitnode(X[:, best_feature], best_thresh)
        Left = self.ExpandTreeGini(X[leftIdxs, :], y[leftIdxs], depth + 1)
        Right = self.ExpandTreeGini(X[rightIdxs, :], y[rightIdxs], depth + 1)
        return TreeNode(best_feature, best_thresh, Bestgain, best_gini, Left, Right)

    def ExpandTree(self, X, y, depth=0):
        sampleCount = X.shape[0]
        featureCount = X.shape[1]
        LableCount = len(np.unique(y))

        if (depth<=self.MaxDepth or LableCount!=1 or 2<sampleCount):
            best_feature, best_thresh, Bestgain, best_entp = self.SplitWithBest(X, y, featureCount)
            if(Bestgain>0):
                leftIdxs, rightIdxs = self.Splitnode(X[:, best_feature], best_thresh)
                Left = self.ExpandTree(X[leftIdxs, :], y[leftIdxs], depth+1)
                Right = self.ExpandTree(X[rightIdxs, :], y[rightIdxs], depth+1)
                return TreeNode(best_feature, best_thresh,Bestgain, best_entp, Left, Right)

        c = Counter(y)
        leaf_Val = c.most_common(2)[0][0]
        return TreeNode(leafVal=leaf_Val)

    def MyEntropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
        # ps = np.bincount(y)/ len(y) 
        # array = []
        # for p in ps:
        #     if p>0:
        #         array.append(p * np.log(p))
        # result_Entp = -1 * np.sum(array)
        # if np.sum(array) == 0:
        #     return 0
        # return result_Entp

    def SplitWithBestGini(self, X, y, feat_idxs):
        best_gini = float('inf')
        split_idx = None
        split_threshold = None
        
        for feat_idx in range(feat_idxs):
            X_column = X[:, feat_idx]
            amountThrs = np.unique(X_column)
            for thr in amountThrs:
                gini = self.calculate_gini_index(y[X_column <= thr])
                if gini < best_gini:
                    best_gini = gini
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold, best_gini, None

    def SplitWithBest(self, X, y, feat_idxs):
        Bestgain = -1
        split_idx = None
        split_threshold = None

        # loop over all the features
        for feat_idx in range(feat_idxs):
            feat_values = X[:, feat_idx]
            possible_thresholds = np.unique(feat_values)
            for thr in possible_thresholds:
                tmp = self.InformationGain(y, feat_values, thr)
                if tmp != 0:
                    gain = tmp[0]
                    entp = tmp[1]
                else:
                    gain = 0
                    entp = 0
                if gain > Bestgain:
                    Bestgain = gain
                    best_entp = entp
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold, Bestgain, best_entp

    def Splitnode(self, dataset, threshold):
        l_idx = np.argwhere(dataset <= threshold)
        dataset_left = l_idx.flatten()
        r_idx = np.argwhere(dataset > threshold)
        dataset_right = r_idx.flatten()
        return dataset_left, dataset_right

    def InformationGain(self, y, X_column, amountThr):
        EntropyParent = self.MyEntropy(y)
        leftIdxs, rightIdxs = self.Splitnode(X_column, amountThr)
        if len(leftIdxs) == 0 or len(rightIdxs) == 0:
            return 0
        # weighted avgerage entropy of children
        y_count = len(y)
        left_count = len(leftIdxs)
        right_count = len(rightIdxs)
        left_entp = self.MyEntropy(y[leftIdxs])
        right_entp = self.MyEntropy(y[rightIdxs])
        childEntropy = (left_count/y_count)*left_entp + (right_count/y_count)*right_entp

        information_gain = EntropyParent - childEntropy
        arr = []
        arr.append(information_gain)
        arr.append(childEntropy)
        return arr

    def TreeTraverse(self, x, node):
        if node.IsLeaf():
            return node.leafVal
        if x[node.feature] <= node.amountThr:
            return self.TreeTraverse(x, node.Left)
        return self.TreeTraverse(x, node.Right)


    def calculate_gini_index(self,y):
        # unique_labels = np.unique(y)
        # gini_index = 0.0
        # for label in unique_labels:
        #     label_probability = np.sum(y == label) / len(y)
        #     gini_index -= label_probability**2
        # return 1-gini_index
        unique_labels = np.unique(y)
        gini_index = 1
        for label in unique_labels:
            label_probability = np.sum(y == label) / len(y)
            gini_index += label_probability * (1 - label_probability)
        return gini_index
            
    def Find_DecisionTree (self,node):
        if node == None:
            return
        if node.IsLeaf():
            print("-leaf: " , node.leafVal)
        else:
            if node.gain is not None:
                print("-feat: ", node.feature, "  -threshold: ", node.amountThr, "  -information gain: ", node.gain, "  -entropy: ",
                node.Entropy)
            else:
                if node.Left is None or node.Right is None:
                    gini_index = np.nan
                else:
                    gini_index = max(self.calculate_gini_index(node.Left), self.calculate_gini_index(node.Right))
                    print("-feat: ", node.feature, "  -threshold: ", node.amountThr, "  -Gini index: ", gini_index)
        self.Find_DecisionTree(node.Left)
        self.Find_DecisionTree(node.Right)

dataset = pd.read_csv("restaurant.csv")
datelen = len(dataset.columns)-1
type_to_number = {
    "French": 1,
    "Thai": 2,
    "Burger": 3,
    "Italian": 4,
}
estimate_to_num={
    "0-10": 1,
    "010-30":2,
    "30-60": 3,
    ">60":4
}
patrons_to_num={
    "some":1,
    "full":2,
    "one":3
}
# map the type column to a number
dataset["type"] = dataset["type"].apply(lambda x: type_to_number[x])
dataset["time"] = dataset["time"].apply(lambda x: estimate_to_num[x])
dataset["patrons"] = dataset["patrons"].apply(lambda x: patrons_to_num[x])
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, datelen].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0 
)

clf_entropy = DecisionTree(MaxDepth=50, criterion='entropy')
clf_entropy.fitRoot(X_train, y_train)

clf_gini = DecisionTree(MaxDepth=50, criterion='gini')
clf_gini.fitRoot(X_train, y_train)
ResArrayEntropy = []
ResArrayGini = []

for xx in X_test:
    res_entropy = clf_entropy.TreeTraverse(xx, clf_entropy.root)
    res_gini = clf_gini.TreeTraverse(xx, clf_gini.root)
    ResArrayEntropy.append(res_entropy)
    ResArrayGini.append(res_gini)

ResultEntropy = np.array(ResArrayEntropy)
ResultGini = np.array(ResArrayGini)

accuracy_entropy = np.sum(y_test == ResultEntropy) / len(y_test)
accuracy_gini = np.sum(y_test == ResultGini) / len(y_test)

print("Accuracy (Entropy):", accuracy_entropy)
print("Entropy-based Decision Tree:")
clf_entropy.Find_DecisionTree(clf_entropy.root)
print("Accuracy (Gini):", accuracy_gini)
print("\nGini-based Decision Tree:")
clf_gini.Find_DecisionTree(clf_gini.root)

