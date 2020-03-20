# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# variables
days_to_consider = 60

# read trained data
data_train = pd.read_csv('./DataSet/Google_Stock_Price_Train.csv')
training_set = data_train.iloc[:, 1:2].values

# feature scalling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set = sc.fit_transform(training_set)

# creating input and output
x_train = []
y_train = []
for i in range(days_to_consider, len(training_set)):
    x_train.append(training_set[i - days_to_consider: i])
    y_train.append(training_set[i])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# test data
data_test = pd.read_csv('./DataSet/Google_Stock_Price_Test.csv')
y_test = data_test.iloc[:, 1:2].values

# all data
data_total = pd.concat((data_train["Open"], data_test["Open"]), axis=0)

# labels
date = data_test.iloc[:, 0].values
date = [s[:-5] for s in date]

# only the last days_to_consider days and y_test
testable_data = data_total[len(data_total) - len(y_test) - days_to_consider:].values
testable_data = testable_data.reshape((-1, 1))
testable_data = sc.transform(testable_data)

# actually x_test
x_test = []
for i in range(days_to_consider, len(testable_data)):
    x_test.append(testable_data[i - days_to_consider:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
