import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
url = "https://raw.githubusercontent.com/kshedden/statswpy-nhanes/master/merged/nhanes_2015_2016.csv"
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Target variable is BPXDI1 (Diastolic Blood Pressure)
# Bin the target variable into categories


def categorize_bpxdi1(value):
    if value < 60:
        return 'low'
    elif value < 80:
        return 'normal'
    else:
        return 'high'


df['BPXDI1_cat'] = df['BPXDI1'].apply(categorize_bpxdi1)

# Drop rows with missing values in target and features
df = df.dropna(subset=['BPXDI1', 'BPXDI1_cat'])
features = df.drop(columns=['SEQN', 'BPXDI1', 'BPXDI1_cat'])

# Drop rows with missing values in features
features = features.dropna()
# Ensure the target variable is also without missing values
df = df.loc[features.index]

# Define features (X) and target (y)
X = features
y = df['BPXDI1_cat']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize the data
normalizer = MinMaxScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Train the SVM model with RBF kernel
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve for each class
fpr = {}
tpr = {}
roc_auc = {}
for i, class_label in enumerate(model.classes_):
    fpr[class_label], tpr[class_label], _ = roc_curve(
        y_test, y_pred_proba[:, i], pos_label=class_label)
    roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])

plt.figure()
for class_label in model.classes_:
    plt.plot(fpr[class_label], tpr[class_label], lw=2,
             label=f'ROC curve for class {class_label} (area = {roc_auc[class_label]:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
