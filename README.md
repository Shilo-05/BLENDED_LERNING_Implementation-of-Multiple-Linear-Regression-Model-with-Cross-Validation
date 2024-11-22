# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

```
Developed by: Oswald SHilo
RegisterNumber: 212223040139
```

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Library Import**:  
   - Import pandas, numpy, and scikit-learn for data manipulation and model building.  

2. **Dataset Loading**:  
   - Load the dataset from `encoded_car_data.csv`.  

3. **Feature and Target Selection**:  
   - Define the features (X) and the target variable (y) for the model.  

4. **Data Splitting**:  
   - Split the dataset into training and testing subsets, maintaining an 80-20 ratio.  

5. **Model Training**:  
   - Train a Multiple Linear Regression model using the training dataset.  

6. **Test Set Evaluation**:  
   - Evaluate the model's predictions on the test set using metrics such as Mean Squared Error (MSE) and R² score.  

7. **Cross-Validation**:  
   - Perform 5-fold cross-validation to calculate MSE and R² values for more robust evaluation.  

8. **Result Presentation**:  
   - Display the evaluation metrics and cross-validation outcomes.  


## Program:
```
# Program to implement the multiple linear regression model for predicting car prices with cross-validation.
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv'
df = pd.read_csv(url)

# Select relevant features and target variable
X = df.drop(columns=['price'])  # All columns except 'price'
y = df['price']  # Target variable

# Split the dataset (not strictly required for cross-validation, but good for validation outside cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("Test Set Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # 5-fold CV
cv_mse = -cv_scores  # Convert negative MSE to positive
print("\nCross-Validation Results:")
print("MSE for each fold:", cv_mse)
print("Mean MSE:", np.mean(cv_mse))
print("Standard Deviation of MSE:", np.std(cv_mse))

# Cross-Validation R-squared
cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nR-squared for each fold:", cv_r2_scores)
print("Mean R-squared:", np.mean(cv_r2_scores))
print("Standard Deviation of R-squared:", np.std(cv_r2_scores))

```

## Output:
<img width="944" alt="Screenshot 2024-11-17 at 7 13 35 PM" src="https://github.com/user-attachments/assets/48c00439-b5b1-4fe6-aad7-f0a7d2583734">


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
