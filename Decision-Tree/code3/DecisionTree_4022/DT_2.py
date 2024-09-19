import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('onlinefraud.csv')

transc_type = {
    "PAYMENT": 1,
    "TRANSFER": 2,
    "CASH_OUT": 3,
    "CASH_IN": 4,
    "DEBIT": 5
}
train["type"] = train["type"].map(transc_type)

train["nameOrig"] = train["nameOrig"].str.replace('C', '', regex=True)
train["nameDest"] = train["nameDest"].str.replace('[MC]', '', regex=True)
train['amount'] = pd.cut(train['amount'], bins=[
                         0, 1000, 10000, 50000, float('Inf')], labels=[0, 1, 2, 3])
label_encoder = LabelEncoder()
for column in ['nameOrig', 'nameDest']:
    train[column] = label_encoder.fit_transform(train[column])


X = train.drop('isFraud', axis=1)
y = train['isFraud']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=42)


dt_entropy.fit(X_train, y_train)
dt_gini.fit(X_train, y_train)

# Predict the test set results
y_pred_entropy = dt_entropy.predict(X_test)
y_pred_gini = dt_gini.predict(X_test)


accuracy_entropy = accuracy_score(y_test, y_pred_entropy) * 100
accuracy_gini = accuracy_score(y_test, y_pred_gini) * 100
print(f"Accuracy (Entropy): {accuracy_entropy:.2f}%")
print(f"Accuracy (Gini): {accuracy_gini:.2f}%")

criteria = ['Entropy', 'Gini']
accuracies = [accuracy_entropy, accuracy_gini]
plt.figure(figsize=(10, 5))
plt.bar(criteria, accuracies, color=['blue', 'green'])
plt.xlabel('Criteria')
plt.ylabel('Accuracy (%)')
plt.title('Decision Tree Accuracy Comparison: Entropy vs Gini')
plt.show()
