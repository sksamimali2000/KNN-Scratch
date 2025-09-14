# 🧠 K-Nearest Neighbors (KNN) Classifier Implementation

A simple implementation of **K-Nearest Neighbors (KNN)** classifier from scratch using **Breast Cancer dataset** from scikit-learn.

This project compares:
- Built-in KNN from `sklearn`
- Manual implementation of KNN algorithm using distance calculation and majority voting.

---

## 🚀 Features

- Trains a KNN classifier using scikit-learn  
- Implements custom KNN prediction from scratch (without using sklearn for prediction)  
- Calculates accuracy of predictions  
- Uses Euclidean distance to compute nearest neighbors

---

## 📊 Dataset

- Uses **Breast Cancer Dataset** (`sklearn.datasets.load_breast_cancer`)  
- Binary classification problem (Malignant vs Benign)  
- 30 numeric features per sample

---

## ⚡ Usage

```python
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Load dataset
dataset = datasets.load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=0)

# Built-in KNN
clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train, Y_train)
print("Sklearn KNN Accuracy:", clf.score(X_test, Y_test))

# Custom KNN Implementation
def predict_one(x_train, y_train, x_test, k):
    distances = []
    for i in range(len(x_train)):
        distance = ((x_train[i, :] - x_test)**2).sum()
        distances.append([distance, i])
    distances = sorted(distances)
    targets = [y_train[distances[i][1]] for i in range(k)]
    return Counter(targets).most_common(1)[0][0]

def predict(x_train, y_train, x_test_data, k):
    predictions = []
    for x_test in x_test_data:
        predictions.append(predict_one(x_train, y_train, x_test, k))
    return predictions

# Predict & Evaluate
y_pred = predict(X_train, Y_train, X_test, 7)
print("Custom KNN Accuracy:", accuracy_score(Y_test, y_pred))
```


✅ Key Outcome

Built-in KNN accuracy measured via clf.score()

Custom implementation of KNN gives comparable accuracy using majority voting on k nearest neighbors.

⚙️ Requirements

Python >= 3.7

scikit-learn

numpy

Install dependencies via:

pip install scikit-learn numpy

📄 License

MIT License

Made with ❤️ by Sk Samim Ali
