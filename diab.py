import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.processing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#reading frfom the dataset
diab_data_set=pd.read_csv('\diabetes_prediction\diabetes.csv')

