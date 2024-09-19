import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


newsgroups = fetch_20newsgroups(subset='all', categories=[
                                'sci.space', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.graphics'])


X, y = newsgroups.data, newsgroups.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear'))


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=newsgroups.target_names, yticklabels=newsgroups.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(14, 7))
plt.plot(y_test[:30], label='Actual Authors', color='blue', marker='o')
plt.plot(y_pred[:30], label='Predicted Authors', color='red', marker='x')
plt.title('Comparison of Actual and Predicted Authors')
plt.xlabel('Sample Index')
plt.ylabel('Author')
plt.legend()
plt.show()
