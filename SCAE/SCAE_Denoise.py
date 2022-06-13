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

def algorithm_1(trace, pt):
    reference_trace = np.zeros(shape=(num_data, input_size))

    for i in range(para.KEY_SIZE):
        ind = np.where(pt==i)[0]
        reference_trace[ind] = np.mean(trace[ind], axis=0)
    
    return reference_trace

# SCAE
def SCAE(trace, pt):
    # get reference traces
    num_data, input_size = np.shape(trace)
    output_size = input_size
    
    label = algorithm_1(trace, pt)
        
    # Network
    inputly = Input(shape=(input_size, ))
    hidden1 = Dense(50, activation='hard_sigmoid')(inputly)
    outputly = Dense(output_size, activation='sigmoid')(hidden1)
            
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