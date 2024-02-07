#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@ Project : WaLeF
@ FileName: mlp.py
@ IDE     : PyCharm
@ Author  : Jimeng Shi
@ Time    : 6/18/22 08:57
"""

from pandas import DataFrame
from pandas import concat
from pandas import concat, read_csv
from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv1D, SimpleRNN, MaxPooling1D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.models import load_model



def mlp_layer(input_shape, mlp_unit1, mlp_unit2, mlp_unit3, mlp_unit4, mlp_unit5, mlp_unit6, mlp_unit7, mlp_unit8, dropout, masked_value, opt_num):
    inputs = keras.Input(shape=(input_shape))
    
    masked_inputs = layers.Masking(mask_value=masked_value)(inputs)
    
    x = Dense(mlp_unit1, activation='relu')(masked_inputs)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit2, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit3, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit4, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit5, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit6, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit7, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit8, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)

    outputs = Dense(opt_num)(x)

    mlp_model = Model(inputs=inputs, outputs=outputs)
    #mlp_model.summary()


    return mlp_model



def mlp_layer_mv(input_shape, mlp_unit1, mlp_unit2, mlp_unit3, mlp_unit4, mlp_unit5, mlp_unit6, mlp_unit7, mlp_unit8, dropout, opt_num, opt_var):
    inputs = keras.Input(shape=(input_shape))
    
    x = Dense(mlp_unit1, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit2, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit3, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit4, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit5, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit6, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit7, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(mlp_unit8, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Reshape((opt_num, -1))(x)
    outputs = Dense(opt_var)(x)
#     outputs = Dense(opt_num)(x)

    mlp_model = Model(inputs=inputs, outputs=outputs)
    #mlp_model.summary()


    return mlp_model
