#importing the library
import numpy as np
import pandas as pd

#reading the dataset and creating the dataframe
dataset = pd.read_csv("data.csv")

#converting all string values to nan
dataset = dataset.convert_objects(convert_numeric=True)

#dividing coloumns between dependent and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

#fitting NaN value with the average values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)
imputer = imputer.fit(x[:,0:13])   
x[:, 0:13] = imputer.transform(x[:, 0:13])
   
#scalng the data on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

#dividing data between test set and training set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2)

#performing logistics regression
from sklearn.linear_model import LogisticRegression
result = LogisticRegression(random_state = 0)

#prediction
result.fit(x_train,y_train)

#percentage accuracy in prediction
y_predict = result.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)