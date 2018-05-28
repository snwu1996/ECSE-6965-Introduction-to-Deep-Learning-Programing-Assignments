"""
Define helper classes and functions for Deep Learning programing assignment 3

Written by Shu-Nong Wu
"""

import numpy as np


NUM_TRAINING_SAMPLES = 50000
NUM_TESTING_SAMPLES = 5000
NUM_CLASSES = 10
width, height, channels = 32, 32, 3


class Data:
    """
    The datasets built into Tensorflow allows you to conveniently call the
    next_batch() function to get the next batch of data.
    This is just a reimplementation of that function.
    """
    def __init__(self, X_data, Y_data):
        self.X_data = X_data
        self.Y_data = Y_data
        self.batch_num = 0
        
    def next_batch(self, batch_size):
        """
        Used for gradient descent when the input data set is too large.
        You can split it up into batches of BATCH_SIZE and iterate through the batches.
        """
        X_batch = self.X_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size,:,:,:]
        Y_batch = self.Y_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size]
        self.batch_num += 1
        return X_batch, Y_batch
    
    def full_batch(self):
        """
        Returns a batch containing all data
        """
        return self.X_data, self.Y_data
    
    def random_batch(self, batch_size):
        """
        Used for stochastic gradient descent.
        Cuts the dataset into batches of BATCH_SIZE and randomly selects one of those batches
        """
        rand_nums = np.random.randint(self.X_data.shape[0], size=(batch_size))
        X_batch = self.X_data[rand_nums,:,:,:]
        Y_batch = self.Y_data[rand_nums]
        return X_batch, Y_batch

    
    def shuffle(self):
        """
        Shuffle the data between every epoch to have faster convergence
        """
        new_X = np.empty(self.X_data.shape, dtype=self.X_data.dtype)
        new_Y = np.empty(self.Y_data.shape, dtype=self.Y_data.dtype)
        perm = np.random.permutation(self.X_data.shape[0])
        for old_idx, new_idx in enumerate(perm):
            new_X[new_idx,:,:,:] = self.X_data[old_idx,:,:,:]
            new_Y[new_idx]       = self.Y_data[old_idx]
        self.X_data = new_X
        self.Y_data = new_Y
        
    def reset(self, shuffle=False):
        self.batch_num = 0
        if shuffle:
            self.shuffle()


def one_hot(number):
    """
    Converts a number into its one_hot representation
    Eg: 4 -> [0,0,0,1,0,0,0,0,0,0]
        2 -> [0,1,0,0,0,0,0,0,0,0]
    This is done because the output of our model is represented as nodes.
    The activation of a node corresponds to some output.
    """
    one_hot_array = np.zeros(NUM_CLASSES, dtype=np.float32)
    one_hot_array[number] = 1
    return one_hot_array