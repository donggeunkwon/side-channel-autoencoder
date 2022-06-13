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

# attack
def attack(trace, pt):
    # SCAE
    def SCAE(trace, pt):
        # get reference traces
        num_data, input_size = np.shape(trace)
        output_size = input_size
        label = np.zeros(shape=(num_data, input_size))
        
        for i in range(para.KEY_SIZE):
            ind = np.where(pt==i)[0]
            label[ind] = np.mean(trace[ind], axis=0)
            
        inputly = Input(shape=(input_size, ))
        hidden1 = Dense(50, activation='sigmoid')(inputly)
        outputly = Dense(output_size, activation='sigmoid')(hidden1)
                
        # Build
        model = Model(inputly, outputly)
        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE), 
                      metrics=['accuracy'])
            
        model.fit(trace, label, 
                  epochs=para.EPOCH, 
                  batch_size=para.BATCH_SIZE, 
                  verbose=0)
        
        label = model.predict(trace)
        
        K.clear_session()
    
        return label
    
    # CPA
    def CPA(trace, pt):
        # corr coef
        def corr(A, B):
            if len(A) != len(B):
                raise(ValueError, "operands could not be broadcast together")
            
            def init():
                a = np.array(A, dtype=np.float64)
                b = np.array(B, dtype=np.float64)        
                
                try :
                    return matlab_corr(a, b)
                except:
                    return np.corrcoef(a, b)[0, 1] # Error
                
            def matlab_corr(A, B): 
                DA = A -  np.mean(A, axis=0)
                DB = B -  np.mean(B, axis=0)
                
                CV = np.dot(DA.T, DB) / np.double(len(A))
            
                VA = np.mean(np.square(DA), axis=0)[:, np.newaxis]
                VB = np.mean(np.square(DB), axis=0)[np.newaxis, :]
            
                return (CV / np.sqrt(np.dot(VA, VB)))
            
            return init()
        
        # Hamming Weight
        def hypohw(pt):
            hypo = np.zeros(shape=(para.KEY_SIZE, np.shape(pt)[0]))
        
            for k in range(para.KEY_SIZE):
                hypo[k] = ([bin(para.SBOX[p ^ k]).count('1') for p in pt])
        
            return np.array(hypo)
        
        hw = hypohw(pt)
        ret_corr = corr(trace, hw.T)
        ret = np.max(np.abs(ret_corr), axis=0)
        guess_key = np.argmax(ret)
        
        return guess_key

    prep_trace = SCAE(trace, pt)
    guess_key = CPA(prep_trace, pt)
    
    return guess_key