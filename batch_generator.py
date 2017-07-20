#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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
        
    def next_batch(self, train_test, hist=False):

        x_batch = []
        y_batch = []
        price_hist_batch = []
        
        for _ in range(self.batch_size):
            
            if train_test == 'train':                
                randint = np.random.randint(0,len(self.train_data_list)-1)                
                
                file_name = self.train_data_list[randint]
                
                x = np.load(file_name)    
                                                
                x_batch.append(x[:3])
                y_batch.append(x[3])               
                
                if hist:
                    price_hist = np.load(file_name.replace('.npy', '_price_history.npy'))                   
                    price_hist_batch.append(price_hist)
                
                    
                
            if train_test == 'test':
                randint = np.random.randint(0,len(self.train_data_list)-1)                
                
                x = np.load(self.train_data_list[randint])                   
                                
                x_batch.append(x[:3])
                y_batch.append(x[3])           
                
                if hist:
                    price_hist = np.load(file_name.replace('.npy', '_price_history.npy'))                   
                    price_hist_batch.append(price_hist)
                
        if hist:
            return np.array(x_batch).reshape([self.batch_size, 3, -1]),np.array(y_batch).reshape([self.batch_size, 1, -1]), np.array(price_hist_batch)
        else:
            return np.array(x_batch).reshape([self.batch_size, 3, -1]),np.array(y_batch).reshape([self.batch_size, 1, -1])
    
                
    
if __name__ == '__main__':
    
    generator=BatchGenerator('/media/sander/samsungssd/tradosaurus/train_data/',
                             '/media/sander/samsungssd/tradosaurus/test_data/',
                             50, 100, 1)
    x,y,h=generator.next_batch('train', True)
    #test = np.load('/media/sander/samsungssd/tradosaurus/train_data/NTAP100.npy')
    
    import matplotlib.pyplot as plt
    from create_features import indexize
    h = [indexize(n) for n in h]
    h = np.array(h)
    plt.plot([0]*98, color='k')
    plt.plot(x[0,0,:], color='g')
    plt.plot(h[0,1:-1], color='b')
    