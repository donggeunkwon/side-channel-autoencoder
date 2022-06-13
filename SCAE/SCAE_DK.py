# -*- coding: utf-8 -*-
"""
@author: Donggeun Kwon (donggeun.kwon@gmail.com)

Cryptographic Algorithm Lab.
Institute of Cyber Security & Privacy (ICSP), Korea University
"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import tensorflow as tf

import hyperparameters as para

import time

def DK_encoding(pt, len=8):
    def arr2barr(arr, l):
        return [list(bin(x)[2:].zfill(l)) for x in arr]
    
    return arr2barr(pt, len)

# SCAE
def SCAE(trace, pt):
    # get reference traces
    num_data, input_size = np.shape(trace)
    output_size = input_size
    
    label = denoising_algorithm(trace)
    dk_input = DK_encoding(pt)

    # Network
    inputly = Input(shape=(input_size, ))
    dk_input = Input(shape=(8, ))
    
    hidden1 = Dense(100, activation='sigmoid')(inputly)
    concat_layer = Concatenate()([hidden1, dk_input])
    hidden2 = Dense(100, activation='sigmoid')(concat_layer)
    outputly = Dense(output_size, activation='sigmoid')(hidden2)
            
    # Build
    model = Model(inputly, outputly)
    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE))
        
    model.fit(trace, label, 
              epochs=para.EPOCH, 
              batch_size=para.BATCH_SIZE, 
              verbose=0)
    
    label = model.predict(trace)
    
    K.clear_session()

    return label