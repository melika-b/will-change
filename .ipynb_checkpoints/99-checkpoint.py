# For linear algebra
import numpy as np
# For data processing
import pandas as pd

#Load the data set
df = pd.read_csv('D:/paper/C/data/5/5data.csv')
#Display the shape of the data set
print('Size of weather data frame is :',df.shape)
#Display data
print(df[1:86])

#3 data preprocessing
# Checking for null values
print(df.count().sort_values())

#remove the variables that are not significant. 
#emove the ‘RISK_MM’ variable
df = df.drop(columns=['date','hours'],axis=1)
print(df.shape)

#Removing null values
df = df.dropna(how='any')
print(df.shape)

#getting rid of outliers
from scipy import stats
z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df= df[(z < 45).all(axis=1)]
print(df.shape)

#To simplify computations we will use only one feature (Humidity3pm) to build the model
 
X = df[['outdoor2017']]
y = df[['GasOffice']]


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
#Calculating the accuracy and the time taken by the classifier
t0=time.time()
#Data Splicing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)

import numpy as np
y_train2=y_train.reshape(-1,1)

from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(training_scores_encoded))




clf_rf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)


#Building the model using the training data set
clf_rf.fit(X_train, y_train)


