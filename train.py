# imports
import numpy as np
import pandas as pd

# preprocessing
# inicializing variables
days_to_consider = 60
batch = 32

# read data set
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

# RNA
# import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# inicialize model
model = Sequential()

# add layers
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1:3])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

# compile RNN
model.compile(optimizer="Adam", loss="mean_squared_error")

# fit RNN
import time

start = time.time()
model.fit(x_train, y_train, batch_size=batch, epochs=256)
fim = time.time()
seg = fim - start
print("Trained in {:.0f}:{:.0f}:{:.0f}".format(seg//3600, seg//60, seg % 60))

# save RNN
model.save("./model")
