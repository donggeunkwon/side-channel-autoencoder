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


def labeling(pt, k, output_size):
    return np.array([np.eye(output_size)[(para.SBOX[int(p) ^ k]) & 1] for p in pt])

def attack(trace, pt):
    # Input/Output Shape 
    input_size = np.shape(trace)[1]
    output_size = 2
    
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