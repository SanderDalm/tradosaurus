#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from six.moves import range
from matplotlib import pyplot as plt


class RNN(object):

    def __init__(self, summary_frequency, num_nodes, num_layers, num_unrollings,
                 batch_generator, input_shape=3, only_retrain_output=False, output_keep_prob=1):

        
        self.batch_generator = batch_generator        
        self.batch_size = self.batch_generator.batch_size
        self.num_unrollings = num_unrollings
        self.num_nodes = num_nodes        
        self.summary_frequency = summary_frequency
        self.input_shape = input_shape
        self.only_retrain_output = only_retrain_output
        self.output_keep_prob = output_keep_prob
        self.session=tf.Session()

        self.minibatch_loss_list = []
        self.loss_list = []

        # Call a basic LSTM/GRU cell from tensorflow module
        cell = tf.nn.rnn_cell.GRUCell(num_nodes)

        cells = [cell]

        # Here we add as many layers as desired
        for i in range(num_layers-1):
          higher_layer_cell = tf.nn.rnn_cell.GRUCell(self.num_nodes)
          cells.append(higher_layer_cell)

        cells = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.output_keep_prob)
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
        self.sample_prediction = logits        

        self.train_labels = list()
        
        logits = tf.reshape(logits, [self.num_unrollings, self.batch_size, 1])
        
        for i in range(self.num_unrollings):
            self.train_labels.append(tf.placeholder(tf.float32, [None, 1]))
                   
        
        self.loss = tf.losses.mean_squared_error(self.train_labels, logits)
       
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
                

    def logprob(self, predictions, labels):

        """Log-probability of the true labels in a predicted batch."""
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


    def train(self, num_steps):

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

                    # The mean loss is an estimate of the loss over the last few batches.
                    print(
                       'Average loss at step %d: %f ' % (step, mean_loss))

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

        feed_dict = dict()
        for i in range(self.num_unrollings):
                    feed_dict[self.train_data[i]] = inputs[:,:,i]                    
        predictions, = self.session.run(
                [self.sample_prediction], feed_dict=feed_dict)

        
        plt.plot(inputs[:,0,1:].reshape([98,1]), color='g', alpha=.4)
        plt.plot(predictions, color='b', alpha=.4)
        
        return predictions


    def plot(self):

        x1=np.array(self.loss_list)
        plt.plot(x1,color='g',alpha=0.4, linewidth=5)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

if __name__ == '__main__':
    
    from batch_generator import BatchGenerator
    train_dir = '/media/sander/samsungssd/tradosaurus/train_data/'
    test_dir = '/media/sander/samsungssd/tradosaurus/test_data/'
    batch_size = 1
    generator = BatchGenerator(train_dir, test_dir, batch_size)

    
    summary_frequency=10
    num_nodes=32
    num_layers=1
    num_unrollings = 99
    batch_generator=generator    
    input_shape = 3
    only_retrain_output=False
    output_keep_prob = 1
    
    nn = RNN(summary_frequency, num_nodes, num_layers, num_unrollings,
                 batch_generator, input_shape, only_retrain_output, output_keep_prob)
    nn.train(1500)
    nn.plot()
    
    x, y = generator.next_batch('test')
    pred = nn.predict(x)