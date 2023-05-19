import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.processing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#reading frfom the dataset
diab_data_set=pd.read_csv('\diabetes_prediction\diabetes.csv')
diab_data_set.head()

x= diab_data_set.drop(columns='Outcome', axis=1)
y=diab_data_set['Outcome']

scaler=StandardScaler()
scaler.fit(x)
standardised_data=scaler.transform(x)
x=standardised_data
y=diab_data_set['Outcome']


x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)
classifier=svm.SVC(kernel='linear')
#training support vector machine
classifier.fit(x_train, y_train)

#getting accuracy score from training data

x_train_prediction=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_test)

#changing dataset into numpy array

input_data=(4,110,922,0,0,37.6,0.191,30)

input_data_as_numpy_array = np.array(input_data)
# reshaping the array
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshape)

print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('the person is non diabetic')
else:
  print('the person is diabetic')
