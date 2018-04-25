import numpy as np
import pandas as pd

dataset = pd.read_csv("data.csv")

#converting all string values to nan
dataset = dataset.convert_objects(convert_numeric=True)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)
imputer = imputer.fit(x[:,0:13])   
x[:, 0:13] = imputer.transform(x[:, 0:13])
   

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.2)

from sklearn.linear_model import LogisticRegression
result = LogisticRegression(random_state = 0)
result.fit(x_train,y_train)

y_predict = result.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)
