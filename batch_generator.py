#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from glob import glob
from six.moves import range

import numpy as np
import tensorflow as tf

class BatchGenerator(object):

    def __init__(self, train_dir, test_dir, batch_size):


        self.train_data_list = glob(train_dir+'/*.npy')
        self.test_data_list = glob(test_dir+'/*.npy')
        self.batch_size = batch_size

    def next_batch(self, train_test):

        x_batch = []
        y_batch = []
        
        for _ in range(self.batch_size):
            
            if train_test == 'train':                
                randint = np.random.randint(0,len(self.train_data_list)-1)
                x = np.load(self.train_data_list[randint])[:,:-1]            
                y = np.load(self.train_data_list[randint])[0][1:]
            if train_test == 'test':
                randint = np.random.randint(0,len(self.test_data_list)-1)
                x = np.load(self.test_data_list[randint])[:,:-1]            
                y = np.load(self.test_data_list[randint])[0][1:]
                
            x_batch.append(x)
            y_batch.append(y)
            
        return np.array(x_batch), np.array(y_batch)
    
if __name__ == '__main__':
    generator=BatchGenerator('/media/sander/samsungssd/tradosaurus/train_data/',
                             '/media/sander/samsungssd/tradosaurus/test_data/',
                             50)
    
    x,y=generator.next_batch('train')
    