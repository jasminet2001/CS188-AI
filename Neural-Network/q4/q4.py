import os
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import re

# Function to load images and labels from the dataset


def load_images(dataset_folder):
    images = []
    labels = []
    for subdir, _, files in os.walk(os.path.join(dataset_folder, "Faces")):
        for file in files:
            filepath = os.path.join(subdir, file)
            label = os.path.splitext(os.path.basename(filepath))[0]
            label = re.sub(r'_\d+', '', label)
            image = imread(filepath, as_gray=True)
            image = resize(image, (50, 50))  # Resize the image
            images.append(image.flatten())  # Flatten the image
            labels.append(label)
    # Convert labels to numeric values
    # label_encoder = LabelEncoder()
    # encoded_labels = label_encoder.fit_transform(labels)
    return np.array(images), np.array(labels)


# Function to load labels from CSV file
def load_labels(csv_file):
    labels_df = pd.read_csv(csv_file)
    return labels_df['label'].values


# Path to the dataset folder and CSV file
dataset_folder = "C://Users//Jasmine//Desktop//TenthSemester//AI//second-project//q4//dataset//Faces"
csv_file = "C://Users//Jasmine//Desktop//TenthSemester//AI//second-project//q4//dataset//Dataset.csv"

# Load images and labels
X, y = load_images(dataset_folder)
labels = load_labels(csv_file)
# print("X\n", X)
# print("Y\n", Y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize MLP classifier with regularization
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)

# Perform cross-validation
cv_scores = cross_val_score(mlp_classifier, X_train, y_train, cv=5)

# Train the classifier
mlp_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# Display each test image with its predicted name
for i in range(len(X_test)):
    plt.figure()
    plt.imshow(X_test[i].reshape(50, 50), cmap='gray')
    plt.title(f"Predicted: {y_pred[i]}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
