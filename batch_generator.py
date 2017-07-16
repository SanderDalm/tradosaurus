#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os

from glob import glob
from six.moves import range

import numpy as np


class BatchGenerator(object):

    def __init__(self, train_dir, test_dir, batch_size, n_history, n_future):


        self.train_data_list = glob(train_dir+'/*.npy')        
        self.test_data_list = glob(test_dir+'/*.npy')
        self.train_data_list = [x for x in self.train_data_list if x.find('hist')==-1]
        self.test_data_list = [x  for x in self.test_data_list if x.find('hist')==-1]
        self.batch_size = batch_size
        self.n_history = n_history
        self.n_future = n_future
        
    def next_batch(self, train_test):

        x_batch = []
        y_batch = []
        
        for _ in range(self.batch_size):
            
            if train_test == 'train':                
                randint = np.random.randint(0,len(self.train_data_list)-1)                
                
                x = np.load(self.train_data_list[randint])                   
                
                x_batch.append(x[:3])
                y_batch.append(x[3])                
                
                    
                
            if train_test == 'test':
                randint = np.random.randint(0,len(self.train_data_list)-1)                
                
                x = np.load(self.train_data_list[randint])                   
                
                x_batch.append(x[:3,:])
                y_batch.append(x[3,:])                
                      
        return np.array(x_batch).reshape([self.batch_size, 3, -1]),np.array(y_batch).reshape([self.batch_size, 1, -1])
    
                
    
if __name__ == '__main__':
    generator=BatchGenerator('/media/sander/samsungssd/tradosaurus/train_data/',
                             '/media/sander/samsungssd/tradosaurus/test_data/',
                             50, 100, 1)
    x,y=generator.next_batch('train')
    
    
    import matplotlib.pyplot as plt
    plt.plot(x[0,0,:])
    