from helpers import keras_helper as kh
from helpers import DataProvider as dp
import numpy as np
import matplotlib.pyplot as plt
from time import  sleep

model = kh.GenerateDense([4, 8, 12, 8, 4, 1], ["relu", "tanh", "relu", "relu", "relu", "relu"], "mse", "adam")
data = dp.GetData("AAPL")

batch_size = 4

for i in range(len(data[-1]) - batch_size):
    print(str(i) + "/" + str(len(data[-1]) - batch_size))
    sleep(0.5)

    X = []
    Y = []
    X.append(data[-1][-batch_size:])
    Y.append(data[-1][-1])
    X = np.array(X)
    Y = np.array(Y)

    kh.FitModel(model, X, Y, 0.9)

plt.plot()
plt.show()

print(model.predict(X))