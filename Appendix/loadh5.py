# -*- coding: utf-8 -*-
"""
@author: Donggeun Kwon (donggeun.kwon@gmail.com)

Cryptographic Algorithm Lab.
Institute of Cyber Security & Privacy (ICSP), Korea University
"""

import os, sys, h5py
import numpy as np

# load measurements
def loadh5(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    
    try:
        in_file  = h5py.File(file_path, "r")
    except:
        print("Error: can't open HDF5 file '%s' ..." % file_path)
        sys.exit(-1)
        
    trace = np.array(in_file['trace'])
    label = np.array(in_file['plaintext'])
    key = np.array(in_file['key'])
        
    return trace, label, key