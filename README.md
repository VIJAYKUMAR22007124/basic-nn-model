# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: B VIJAY KUMAR
### Register Number: 212222230173

```

from tensorflow import keras
from keras import models

from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('DS').sheet1


rows = worksheet.get_all_values()


df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df.head()

df

X = df.iloc[: , : -1].values
y = df.iloc[: , -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = MinMaxScaler()

scaler.fit(X_train.reshape(-1,1))

X_train1 = scaler.transform(X_train.reshape(-1,1))

n = models.Sequential([layers.Dense(units = 1 , activation = 'relu' , input_shape = [1]),
                       layers.Dense(units = 3, activation = 'relu'),
                       layers.Dense(units = 3 , activation = 'relu' ),
                       layers.Dense(units = 3 , activation = 'relu'),
                       layers.Dense(units = 1)
])

n.summary()

n.compile(optimizer = 'rmsprop' , loss = 'mse')

n.fit(X_train1 , y_train , epochs = 500)

loss = pd.DataFrame(n.history.history)

loss.plot()

X_test1 = scaler.fit_transform(X_test)

n.evaluate(X_test1 , y_test)

i = [[30]]

i = scaler.fit_transform(i)

n.predict(i)

```
## Dataset Information

![image](https://github.com/user-attachments/assets/bb2128c6-cda9-421b-98ef-620eca6f80bc)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/ff2e907e-8b0b-45d3-b969-f5ee18fd9282)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/19236cec-4ec6-4d57-b272-a39908850eb6)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/b76c7fe0-2b85-4176-8f39-4df9aaf43113)


## RESULT

Thus, the linear regressin network is built and implemented to predict the given input .
