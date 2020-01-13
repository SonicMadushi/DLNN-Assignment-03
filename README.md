# DLNN-Assignment-03

This Assignment materials contains, historical data of Stock values of Microsoft Coperation. You have to use the ```1.0 LSTM for Stock Market Value Prediction .ipynb```
and ```1.1 LSTM for Stock Market Value Prediction .ipynb with necessary modification``` to apply the 1D Convolutional Network provided below for the dataset.

## Dataset

[Download from www.nasdaq.com](https://www.nasdaq.com/market-activity/stocks/msft/historical)

## Instructions

1. Modify the ```1.0 LSTM for Stock Market Value Prediction .ipynb``` read the ```MSFT_data.csv```
2. ```MSFT_data.csv``` contains ```2516``` entities, split them as ```2000``` for training and ```516``` for testing
3. Apply the following 1D Convolutional Neural Network and observe the results.

```python

from keras.models import Sequential
from keras.layers import Conv1D,MaxPooling1D,Dense,Dropout,Activation,Flatten

model=Sequential()

model.add(Conv1D(filters=256,kernel_size=3,input_shape=(data.shape[1:])))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=128,kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=64,kernel_size=3))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.summary()

```
4. Do necessary modifications to improve the performances (maximize the Accuracy, minimize the loss and minimize the overfitting) if required.
