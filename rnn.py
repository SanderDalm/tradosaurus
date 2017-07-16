#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range
from matplotlib import pyplot as plt


class RNN(object):

    def __init__(self, summary_frequency, num_nodes, num_layers, num_unrollings, n_future,
                 batch_generator, input_shape=3, only_retrain_output=False, output_keep_prob=1,
                 cell=tf.nn.rnn_cell.LSTMCell):
        
        self.batch_generator = batch_generator        
        self.batch_size = self.batch_generator.batch_size
        self.num_unrollings = num_unrollings
        self.n_future = n_future
        self.num_nodes = num_nodes        
        self.summary_frequency = summary_frequency
        self.input_shape = input_shape
        self.only_retrain_output = only_retrain_output
        self.output_keep_prob = output_keep_prob
        self.session=tf.Session()

        self.minibatch_loss_list = []
        self.loss_list = []
        self.val_loss_list = []
        
        self.is_training = True
        cells = [cell(self.num_nodes) for _ in range(num_layers)]
        
        if self.is_training:
            cells = [tf.nn.rnn_cell.DropoutWrapper(cell_, output_keep_prob = self.output_keep_prob)
                      for cell_ in cells]

        
        # These layers are combined into a conventient MultiRNNCell object
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # Read input data        
        self.train_data = list()
        for _ in range(self.num_unrollings):
          self.train_data.append(
            tf.placeholder(tf.float32, shape=[None, self.input_shape]))

        
        # Feed the data to the RNN model
        outputs = tf.Variable(np.zeros([self.num_unrollings, self.batch_size, 1]))
        outputs, self.state = tf.nn.static_rnn(multi_cell, self.train_data, dtype=tf.float32)
                
        # Classifier. For training, we remove the last output, as it has no label.
        # The last output is only used for prediction purposes during sampling.
        self.w = tf.Variable(tf.truncated_normal([self.num_nodes, 1], -0.1, 0.1), name='output_w')
        self.b = tf.Variable(tf.zeros([1]), name='output_b')
                
        logits = tf.matmul(tf.concat(axis=0,values=outputs), self.w) + self.b        
        logits = tf.reshape(logits, [self.num_unrollings, -1, 1])

        self.sample_prediction = logits        

        self.train_labels = list()
        for i in range(self.num_unrollings):
            self.train_labels.append(tf.placeholder(tf.float32, [None, 1]))
                           
        self.loss = tf.losses.mean_squared_error(self.train_labels, logits)
        #self.loss =  self.loss = tf.reduce_mean(
        #  tf.nn.softmax_cross_entropy_with_logits(
        #    logits=logits, labels=self.train_labels))
       
        # Optimizer.
        if self.only_retrain_output:
            self.optimizer = tf.train.AdamOptimizer().minimize(loss=self.loss, var_list=[self.w, self.b])
        else:            
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)


        # Train prediction. We keep this to keep track of the model's progress.
        self.train_prediction = tf.nn.softmax(logits)        
        self.session=tf.Session()
        with self.session.as_default():
            init_op = tf.global_variables_initializer()
            self.session.run(init_op)
                


    def train(self, num_steps):
        
        self.is_training = True

        with self.session.as_default():

            mean_loss=0

            for step in range(num_steps):

                feed_dict = dict()
                
                x_batch, y_batch = self.batch_generator.next_batch('train')
                y_batch = y_batch.reshape([self.batch_generator.batch_size, self.num_unrollings, 1])
                for i in range(self.num_unrollings):  
                    
                    feed_dict[self.train_data[i]] = x_batch[:,:,i]                    
                    feed_dict[self.train_labels[i]] = y_batch[:,i]
                
                
                _, l, predictions, = self.session.run(
                            [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)

                mean_loss += l

                if step % self.summary_frequency == 0:

                    if step > 0:
                        mean_loss = mean_loss / self.summary_frequency
                        self.loss_list.append(mean_loss)
                        val_loss = self.get_val_loss()
                        self.val_loss_list.append(val_loss)

                        # The mean loss is an estimate of the loss over the last few batches.
                        print('       Average train loss at step %d: %f ' % (step, mean_loss))
                        print('Average val loss at step %d: %f ' % (step, val_loss))

                    mean_loss=0


    def return_output_weights(self):

        feed_dict=dict()
        x_batch, y_batch = self.batch_generator._next()
        for i in range(self.num_unrollings + 1):
            feed_dict[self.train_data[i]] = x_batch[i]

        return self.session.run([self.w], feed_dict=feed_dict)[-1]


    def create_restore_dict(self):

        variable_names = [v for v in tf.trainable_variables()]
        variable_handles = [v.name for v in variable_names]
        restore_dict = dict(zip(variable_handles, variable_names))
        restore_dict.pop('Variable:0')
        restore_dict.pop('output_w:0')
        restore_dict.pop('output_b:0')

        return restore_dict


    def save(self, checkpointname, full_model=True):

        self.saver = tf.train.Saver()
        if full_model == False:
            restore_dict = self.create_restore_dict()
            with self.session.as_default():
                self.saver = tf.train.Saver(restore_dict)

        self.saver.save(self.session, checkpointname)
        print('Model saved')


    def load(self, checkpointname, full_model=True):

        self.saver = tf.train.Saver()
        if full_model == False:
            restore_dict = self.create_restore_dict()
            with self.session.as_default():
                self.saver = tf.train.Saver(restore_dict)

        self.saver.restore(self.session, checkpointname)
        print('Model restored')


    def predict(self, inputs):

        self.is_training = False
        
        feed_dict = dict()
        for i in range(self.num_unrollings):
                    feed_dict[self.train_data[i]] = inputs[:,:,i]                    
        predictions, = self.session.run(
                [self.sample_prediction], feed_dict=feed_dict)
                
        return predictions


    def get_val_loss(self):
        
        x_val, y_val = self.batch_generator.next_batch('test')
        
        feed_dict = dict()
        labels = list()
        
        for i in range(self.num_unrollings):
                    feed_dict[self.train_data[i]] = x_val[:,:,i]  
                    labels.append(y_val[:,:,i])
                                        
        pred, = self.session.run(
                [self.sample_prediction], feed_dict=feed_dict)
        
        labels = np.array(labels)
        
        return np.mean(np.square(labels-pred))
         

    def plot_loss(self):

        x1=np.array(self.loss_list)
        x2=np.array(self.val_loss_list)
        plt.plot(x1,color='g',alpha=0.4, linewidth=5)
        plt.plot(x2,color='r',alpha=0.4, linewidth=5)
        plt.xlabel('Iterations')
        plt.legend(['train_loss', 'val_loss'])
        plt.show()

if __name__ == '__main__':
    
    from batch_generator import BatchGenerator
    train_dir = '/media/sander/samsungssd/tradosaurus/train_data/'
    test_dir = '/media/sander/samsungssd/tradosaurus/test_data/'
    batch_size = 32
    n_future = 1
    generator = BatchGenerator(train_dir, test_dir, batch_size, n_future)

    
    summary_frequency=10
    num_nodes=16
    num_layers=3
    num_unrollings = 100-n_future
    batch_generator=generator    
    input_shape = 3
    only_retrain_output=False
    output_keep_prob = 1
    cell=tf.nn.rnn_cell.LSTMCell
    
    nn = RNN(summary_frequency, num_nodes, num_layers, num_unrollings, n_future,
                 batch_generator, input_shape, only_retrain_output, output_keep_prob,
                 cell)
    nn.train(5000)
    nn.plot()
       
    x, y = generator.next_batch('train')
    x, y = x[0], y[0]
    y = y.reshape([100-n_future, 1])
    x = x.reshape([1, 3, 100-n_future])
    
    plt.plot(x[0,0,:])
    plt.plot(y)
    
    pred = nn.predict(x)
    
    #pred_plot = np.concatenate([np.zeros(n_future).reshape([n_future,1]), predictions])
    
    plt.scatter(pred, y)
    
    plt.plot(x[0,0,:], color='g', alpha=.4)
    plt.plot(y, color='r', alpha=.4)
    plt.plot(pred, color='b', alpha=.4)
        
        
    