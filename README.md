# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import StandardScaler to standardize the features and target variable.
2. Implement a function for linear regression with gradient descent, taking input features X1, target variable y, learning rate, and number of iterations as parameters.
3. Add a column of ones to feature matrix X for the intercept term and initialize theta (parameters) with zeros.
4. Iterate through the specified number of iterations, computing predictions, errors, and updating theta using gradient descent.
5. Read the dataset into a DataFrame, assuming the last column as the target variable 'y' and preceding columns as features 'X'.
6. Standardize the features and target variable using StandardScaler, learn model parameters using linear_regression function, and predict the target value for a new data point after scaling it.  

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: SARGURU K
RegisterNumber: 212222230134
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  x = np.c_[np.ones(len(x1)),x1]
  theta = np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (x).dot(theta).reshape(-1,1)
    errors = (predictions - y).reshape(-1,1)
    theta = learning_rate * (1/len(x1))*x.T.dot(errors)
  return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head())
x = (data.iloc[1:, :-2].values)
print(X)
x1=x.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_Scaled = scaler.fit_transform(x1)
y1_Scaled = scaler.fit_transform(y)
print(x1_Scaled)
print(y1_Scaled)
theta = linear_regression(x1_Scaled, y1_Scaled )

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")

```

## Output:
![Screenshot 2024-03-15 212528](https://github.com/DEVADARSHAN2/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119432150/183055b1-66d2-4729-97de-f55c7694a6a2)

![Screenshot 2024-03-15 212550](https://github.com/DEVADARSHAN2/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119432150/53890340-1a3a-4994-b682-0c98464b46fe)

![Screenshot 2024-03-15 213143](https://github.com/DEVADARSHAN2/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119432150/1dc70c48-50fb-464b-8432-dd9c6b7c61e7)

![Screenshot 2024-03-15 213212](https://github.com/DEVADARSHAN2/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119432150/3bd84f54-c78a-4aba-96a7-f8a61cefc5c9)
### Predicted Value:
![Screenshot 2024-04-04 135939](https://github.com/DEVADARSHAN2/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119432150/a9d73e09-3538-4a84-a0f9-cd5f2f9574a9)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
