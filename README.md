# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Start the program.

2. Data preprocessing

3. Cleanse data,handle missing values,encode categorical variables.

4. Model Training:Fit logistic regression model on preprocessed data.

5. Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.

6. Prediction: Predict placement status for new student data using trained model.

7. End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Jeevan Vishal.G.D
RegisterNumber:  212224240062
*/


# Logistic Regression using Gradient Descent
# Placement Prediction

import pandas as pd
import numpy as np

# -----------------------------
# LOAD DATASET
# -----------------------------
data = pd.read_csv(r"C:\Users\admin\Downloads\Placement_Data.csv")

# Remove unwanted columns
data1 = data.drop(["sl_no", "salary"], axis=1)

# -----------------------------
# LABEL ENCODING
# -----------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# -----------------------------
# SPLIT FEATURES AND TARGET
# -----------------------------
X = data1.iloc[:, :-1].values
Y = data1["status"].values

# -----------------------------
# FEATURE SCALING
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

# -----------------------------
# INITIALIZE PARAMETERS
# -----------------------------
theta = np.random.randn(X.shape[1])

alpha = 0.01
iterations = 1000

# -----------------------------
# SIGMOID FUNCTION
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# LOSS FUNCTION
# -----------------------------
def loss(theta, X, Y):

    h = sigmoid(X.dot(theta))

    return -np.sum(Y * np.log(h + 1e-15) +
                   (1 - Y) * np.log(1 - h + 1e-15)) / len(Y)

# -----------------------------
# GRADIENT DESCENT FUNCTION
# -----------------------------
def gradient_descent(theta, X, Y, alpha, iterations):

    m = len(Y)

    for i in range(iterations):

        h = sigmoid(X.dot(theta))

        gradient = X.T.dot(h - Y) / m

        theta = theta - alpha * gradient

    return theta

# -----------------------------
# TRAIN MODEL
# -----------------------------
theta = gradient_descent(theta, X, Y,
                         alpha, iterations)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(theta, X):

    h = sigmoid(X.dot(theta))

    return np.where(h >= 0.5, 1, 0)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
y_pred = predict(theta, X)

accuracy = np.mean(y_pred == Y)

# -----------------------------
# DISPLAY OUTPUT
# -----------------------------
print("Accuracy:", accuracy)

print("\nPredicted:")
print(y_pred)

print("\nActual:")
print(Y)

# -----------------------------
# NEW DATA PREDICTION
# -----------------------------
xnew = np.array([[0, 87, 0, 95, 0, 2,
                  78, 2, 0, 0, 1, 0]])

xnew = scaler.transform(xnew)

y_prednew = predict(theta, xnew)

print("\nPredicted Result:", y_prednew)


```

## Output:


<img width="1078" height="726" alt="image" src="https://github.com/user-attachments/assets/a3ec386a-42e7-40ff-b011-416490189f6f" />

<img width="957" height="752" alt="image" src="https://github.com/user-attachments/assets/22331907-ad21-4ea7-86a7-f6e1e536e1a0" />

<img width="851" height="626" alt="image" src="https://github.com/user-attachments/assets/ca405a99-7053-4e5b-8475-ad130879196a" />

<img width="705" height="316" alt="image" src="https://github.com/user-attachments/assets/92df2ad9-15a3-4d70-8a9a-de7a6b7b9bc5" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

