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
    
    # DDLA
    def DDLA(trace, pt):
        # Input/Output Shape 
        input_size = np.shape(trace)[1]
        output_size = 2 # MSB or LSB
        
        # MLP_{exp}
        inputly = Input(shape=(input_size, ))
        hidden1 = Dense(20, activation='relu')(inputly)
        hidden2 = Dense(10, activation='relu')(hidden1)
        outputly = Dense(output_size, activation='softmax')(hidden2)
                
        # Build
        model = Model(inputly, outputly)
        model.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE), 
                    metrics=['accuracy'])
        model.save('./reference_model.h5')
        # model.summary()
        
        history = []
        
        for k in range(para.KEY_SIZE):
            # Labeling
            label = labeling(pt, k, output_size)
            
            # MLP_{exp}
            model = load_model('./reference_model.h5')
            model.compile(loss='categorical_crossentropy',
                        optimizer=tf.keras.optimizers.Adam(lr=para.LEARNING_RATE), 
                        metrics=['accuracy']) 
            
            # Train
            hist = model.fit(trace, label, 
                            epochs=para.EPOCH, 
                            batch_size=para.BATCH_SIZE, 
                            verbose=0)
            
            # save history
            history.append(hist.history['accuracy'])
            # np.save('./history_acc_'+str(k), hist.history['acc'])
            # np.save('./history_loss_'+str(k), hist.history['loss'])
        
            K.clear_session()

        hist  = np.array(history)
        ret = np.max(hist.T, axis=0)
        guess_key = np.argmax(ret)

        return guess_key

    prep_trace = SCAE(trace, pt)
    guess_key = DDLA(prep_trace, pt)
    
    return guess_key