# inicializing variables
batch = 32


def train(x_train, y_train):
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
    print("Trained in {:.0f}:{:.0f}:{:.0f}".format(seg // 3600, seg // 60, seg % 60))

    # save RNN
    model.save("./model")
