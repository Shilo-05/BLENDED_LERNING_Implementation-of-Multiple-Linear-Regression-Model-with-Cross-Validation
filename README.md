# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### Import Libraries:
Import necessary libraries like pandas, numpy for data handling, sklearn for model building and evaluation, and matplotlib for visualization.

#### Load the Dataset:
Use pandas.read_csv() to load the car sales dataset into the environment.

#### Data Preprocessing:
Handle missing values, remove irrelevant columns, and encode categorical variables using techniques like one-hot encoding.

#### Build and Train the Model:
Initialize and train the LinearRegression() model from sklearn using .fit() on the training data to learn the relationship between features and the target (car price).

#### Evaluate and Visualize:
Use metrics like MSE and R² for evaluation. Visualize the predicted vs actual car prices with a scatter plot, drawing a reference line for comparison.



## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Oswald Shilo
RegisterNumber: 212223040139 
*/
```

```
# Importing necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset from the given URL
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Data preprocessing
# Dropping unnecessary columns: 'CarName' and 'car_ID' as they are not useful for prediction
data = data.drop(['CarName', 'car_ID'], axis=1)

# Convert categorical variables into dummy/indicator variables using one-hot encoding
# drop_first=True avoids the dummy variable trap
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features (X) and the target variable (y)
X = data.drop('price', axis=1)  # Features (all columns except 'price')
y = data['price']               # Target variable ('price')

# Splitting the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the multiple linear regression model
model = LinearRegression()

# Fitting the model on the training data to learn relationships between features and price
model.fit(X_train, y_train)

# Evaluating model performance using cross-validation (5-fold)
# This will provide an estimate of how well the model generalizes to unseen data
cv_scores = cross_val_score(model, X, y, cv=5)

# Printing the cross-validation scores for each fold
print("Cross-validation scores:", cv_scores)

# Calculating and printing the mean cross-validation score (average performance across all folds)
print("Mean cross-validation score:", cv_scores.mean())

# Print the intercept and coefficients of the trained model
print("Intercept:", model.intercept_)   # Intercept term (constant)
print("Coefficients:", model.coef_)     # Coefficients for each feature

# Making predictions on the test data
predictions = model.predict(X_test)

# ---- Visualization ----
# Plotting the actual prices vs predicted prices to assess model performance
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")  # Label for the x-axis (true prices)
plt.ylabel("Predicted Prices")  # Label for the y-axis (predicted prices by the model)
plt.title("Actual vs Predicted Prices")  # Title of the plot

# Plotting a red line showing perfect prediction (y=x), for visual reference
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

# Display the plot
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/15f9d4fe-d82a-4a63-b981-6f96d9846c4d)

![image](https://github.com/user-attachments/assets/9003c0de-8055-4d52-9204-0f28f0129fb1)



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
