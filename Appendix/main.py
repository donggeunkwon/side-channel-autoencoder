# -*- coding: utf-8 -*-
"""
@author: Donggeun Kwon (donggeun.kwon@gmail.com)

Cryptographic Algorithm Lab.
Institute of Cyber Security & Privacy (ICSP), Korea University
"""

import time
import numpy as np

from loadh5 import loadh5

mode = ['DDLA', 'SCAE_CPA']
MODE_USED = mode[0] # choose the method
exec('import ' + MODE_USED + ' as DLSCA')

def attack(trace, label):
    DLSCA.attack(trace, label)


if __name__ == '__main__':
    print(MODE_USED + 'based SCA start...')

    # Data Load
    start = time.time() # Start
    
    trace, pt, key = loadh5(Dataset_filename)
    # trace = np.double(trace)
    trace = (trace - np.mean(trace))
    trace = (trace) / (np.max(np.abs(trace)))
    
    ret = attack(trace, pt)
    print(MODE_USED + " exec time: ", time.time() - start) #End
