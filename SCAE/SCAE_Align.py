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

def corr(A, B):
    if len(A) != len(B):
        raise(ValueError, "operands could not be broadcast together")
              
    def _corr(A, B): 
        a = np.array(A, dtype=np.float64)
        b = np.array(B, dtype=np.float64)  
        
        DA = A -  np.mean(A, axis=0)
        DB = B -  np.mean(B, axis=0)
        
        CV = np.dot(DA.T, DB) / np.double(len(A))
    
        VA = np.mean(np.square(DA), axis=0)[:, np.newaxis]
        VB = np.mean(np.square(DB), axis=0)[np.newaxis, :]
    
        return (CV / np.sqrt(np.dot(VA, VB)))
    
    return _corr(a, b)

def algorithm_2(trace, pt):
    reference_trace = np.zeros(shape=(num_data, input_size))

    ref_t0 = trace[np.where(pt==0)[0][0]][:]
    
    for i in range(1, para.KEY_SIZE):
        ind = np.where(pt==i)[0]
        ret_corr = corr(trace[ind], ref_t0)
        refer_t1 = trace[ind[np.argmax(ret_corr)]][:]
        reference_trace[ind] = refer_t1
    
    return reference_trace

# SCAE
def SCAE(trace, pt):
    # get reference traces
    num_data, input_size = np.shape(trace)
    output_size = input_size
    
    label = algorithm_2(trace, pt)
        
    # Network
    inputly = Input(shape=(input_size, ))
    # Conv Layers 
    '''
    removed
    '''
    hidden1 = Dense(100, activation='hard_sigmoid')(inputly)
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