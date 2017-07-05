import os
import time
import numpy as np
import matplotlib.pyplot as plt


def get_reward(vector, action):

    reward = vector[-1] - vector[0]

    return np.multiply(action, reward)


def get_ranks(array):

    temp = array.argsort(axis=0)
    ranks = np.empty(len(array))
    ranks[temp] = np.arange(len(array))
    return ranks


def weighted_average(results, num_layers):

    updates = []

    for index in range(num_layers):

        shape = results[0][1][index].shape

        total_factor = 0
        new_layer = np.zeros(shape=shape)

        for r in results:
            factor = r[0]
            total_factor += factor
            weight_list = r[1]
            layer = weight_list[index]
            update = np.multiply(layer, factor)
            new_layer = np.add(new_layer, update)

        new_layer = np.divide(new_layer, total_factor)
        updates.append(new_layer)


    return updates



class tradosaurus(object):

    def __init__(self, num_nodes_hidden, gen_size, learning_rate, decay, num_features, mean, sd):

        self.mean = mean
        self.sd = sd
        self.num_features = num_features
        self.num_nodes_hidden = num_nodes_hidden
        self.gen_size = gen_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.input_size = self.num_features
        self.output_size = 1
        self.weight_list = []

        # Connect inputs
        if len(self.num_nodes_hidden) > 0:

            self.weight_list.append(np.zeros([self.input_size, self.num_nodes_hidden[0]]))

            # Create hidden layers
            for index, num_nodes in enumerate(self.num_nodes_hidden):
                if index+1 < len(self.num_nodes_hidden):
                    self.weight_list.append(np.zeros([self.num_nodes_hidden[index], self.num_nodes_hidden[index+1]]))
                else:
                    self.weight_list.append(np.zeros([self.num_nodes_hidden[index], self.output_size]))

        else:
            self.weight_list = [np.zeros([self.input_size,self.output_size])]


    def trade(self, input_vectors, weight_list):

        # Generate trade action for each input vector
        actions = []
        reward = 0
        for vector in input_vectors:

            x = vector[:self.num_features]
            y = vector[self.num_features:]

            for layer in weight_list:

                x = np.tanh(np.matmul(x, layer))

            reward += get_reward(y, x)
            actions.append(x)

        reward = reward/len(input_vectors)
        return actions, reward


    def run_epoch(self, traindata):

        total_reward = []
        total_weights = []

        for i in range(self.gen_size):

            # Copy and perturb weights
            temp_weight_list = []
            for index, layer in enumerate(self.weight_list):
                temp_weight_list.append(np.add(layer, np.random.normal(self.mean, self.sd, [layer.shape[0], layer.shape[1]])))

            # Calculate reward
            _, reward = self.trade(traindata, temp_weight_list)
            total_reward.append(reward)
            total_weights.append(temp_weight_list)

        # Get ranks and zip rewards and weights
        total_reward = np.array(total_reward)
        reward_ranks = get_ranks(total_reward)
        results = zip(reward_ranks, total_weights)

        new_weights = weighted_average(results, len(self.num_nodes_hidden)+1)

        # Update weights if epoch contains improvement
        for index, layer in enumerate(self.weight_list):
            #self.weight_list[index] = layer  + np.multiply(new_weights[index], self.learning_rate)
            self.weight_list[index] = new_weights[index]


    def evolve(self, num_epochs, traindata, testdata):

        train_rewards = []
        test_rewards = []

        start = time.time()
        for epoch in range(num_epochs):
            print epoch
            self.sd = self.sd*self.decay
            self.run_epoch(traindata)
            action, train_reward = self.trade(traindata, self.weight_list)
            action, test_reward = self.trade(testdata, self.weight_list)
            train_rewards.append(train_reward)
            test_rewards.append(test_reward)
        end = time.time()
        print 'Train time: {} seconds'.format(end-start)
        print 'End reward train: {}'.format(train_reward)
        print 'End reward test: {}'.format(test_reward)
        plt.plot(train_rewards, c='g')
        plt.plot(test_rewards, c='r')
        plt.legend(['train', 'test'])

    def return_weights(self):

        return self.weight_list


class retardosaurus(tradosaurus):

    def trade(self, input_vectors, weights):

        # Generate trade action for each input vector
        actions = []
        reward = 0
        for vector in input_vectors:

            action = np.random.uniform(0,1)
            reward += get_reward(vector, action)
            actions.append(action)

        return actions, reward

