import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

data = pdr.get_data_yahoo("AAPL")
arr_data = []
for j in range(len(data.columns)):
    arr_data.append([])
    for i in range(data.shape[0]):
        arr_data[-1].append(data.iat[i,j])
for i in range(len(data.columns)):
    if i != 4:
        plt.plot(arr_data[i], label=data.columns[i])
plt.legend()
plt.show()