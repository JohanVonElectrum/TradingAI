from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

def GenerateDense(topology, activations, loss, optimizer):
    model = Sequential()

    model.add(Dropout(0.1))
    for i in range(len(topology)):
        model.add(Dense(units=topology[i], activation=activations[i]))

    model.compile(loss=loss, optimizer=optimizer, metrics=["acc", "mse"])

    return model

def FitModel(model, X, Y, min_acc):
    acc = []
    loss = []
    hist = model.fit(np.array(X), np.array(Y), epochs=100, validation_data=(X, Y))
    acc.append(hist.history.get("acc"))
    loss.append(hist.history.get("loss"))
    while np.amax(acc) < min_acc and np.amin(loss) > 0.01:
        hist = model.fit(np.array(X), np.array(Y), epochs=100, validation_data=(X, Y))
        acc.append(hist.history.get("acc"))
        loss = hist.history.get("loss")
    return np.array(loss).flatten()