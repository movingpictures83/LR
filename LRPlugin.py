#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from pandas import concat
from pandas import read_csv
from helper import series_to_supervised, stage_series_to_supervised
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


# In[2]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import PyPluMA

class LRPlugin:
 def input(self, inputfile):
  self.dataset = pd.read_csv(inputfile, index_col=0)
 def run(self):
     pass
 def output(self, outputfile):
  self.dataset.fillna(0, inplace=True)
  data = self.dataset[['MEAN_RAIN', 'WS_S4',
                'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
                'PUMP_S25B', 'PUMP_S26',
                #'FLOW_S25A', 'FLOW_S25B', 'FLOW_S26', 
                'HWS_S25A', 'HWS_S25B', 'HWS_S26',
                'WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
  x = self.dataset[['MEAN_RAIN', 'WS_S4',
            'GATE_S25A', 'GATE_S25B', 'GATE_S25B2', 'GATE_S26_1', 'GATE_S26_2',
            'PUMP_S25B', 'PUMP_S26',
            'HWS_S25A', 'HWS_S25B', 'HWS_S26',
            ]]  # 'TWS_S25A', 'TWS_S25B', 'TWS_S26'
  y = self.dataset[['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']]
  x_np = x.values
  y_np = y.values

  n_train_hours = int(len(x_np)*0.8)
  train_x = x_np[:n_train_hours, :]    # 0 column is the rainfall to measure heavy/medium/light
  test_x = x_np[n_train_hours:, :]

  train_y = y_np[:n_train_hours, :]
  test_y = y_np[n_train_hours:, :]
  model = keras.Sequential()
  model.add(layers.Input(shape=(12,)))  # Input layer for two features
  model.add(layers.Dense(units=4))  # Output layer with two units (for two output variables)
  model.summary()
  lr = 0.00001
  EPOCHS = 20
  model.compile(
              optimizer=Adam(learning_rate=lr, decay=lr/EPOCHS),
              loss='mse',
              metrics=['mae'])

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=500)
  mc = ModelCheckpoint(PyPluMA.prefix()+"/saved_model/lr.h5", monitor='val_mae', mode='min', verbose=2, save_best_only=True)


  history = model.fit(train_x, train_y,
                    batch_size=512,
                    epochs=EPOCHS,
                    validation_data=(test_x, test_y),
                    verbose=2,
                    shuffle=False,
                               callbacks=[es, mc])

  plt.rcParams["figure.figsize"] = (8, 6)
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('Epoch', fontsize=16)
  plt.ylabel('Loss', fontsize=16)
  plt.legend(fontsize=14)
  plt.title("Training loss vs Testing loss", fontsize=18)
  plt.show()

  from tensorflow.keras.models import load_model

  model_load = load_model(PyPluMA.prefix()+"/saved_model/lr.h5")

  yhat = model_load.predict(test_x)
  inv_yhat = yhat
  inv_y = test_y

  inv_yhat = pd.DataFrame(inv_yhat)
  inv_y = pd.DataFrame(inv_y)

  from sklearn.metrics import mean_squared_error as mse
  from sklearn.metrics import mean_absolute_error as mae

  print('MAE = {}'.format(float("{:.6f}".format(mae(inv_y, inv_yhat)))))
  print('RMSE = {}'.format(float("{:.6f}".format(sqrt(mse(inv_y, inv_yhat))))))

  inv_yhat.to_csv(outputfile+"/inv_yhat_lr.csv")

