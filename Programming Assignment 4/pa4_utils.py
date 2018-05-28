import numpy as np


NUM_TRAINING_SAMPLES = 5600
NUM_TESTING_SAMPLE = 2400
SEQ_LEN = 10
GT_SHAPE = (7,2) #Ground truth shape
width, height, channels = 64, 64, 3

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
        X_batch = self.X_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size,:,:,:,:]
        Y_batch = self.Y_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size,:,:,:]
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
        X_batch = self.X_data[rand_nums,:,:,:,:]
        Y_batch = self.Y_data[rand_nums,:,:,:]
        return X_batch, Y_batch

    def shuffle(self):
        """
        Shuffle the data between every epoch to have faster convergence
        """
        new_X = np.empty(self.X_data.shape, dtype=self.X_data.dtype)
        new_Y = np.empty(self.Y_data.shape, dtype=self.Y_data.dtype)
        perm = np.random.permutation(self.X_data.shape[0])
        for old_idx, new_idx in enumerate(perm):
            new_X[new_idx,:,:,:,:] = self.X_data[old_idx,:,:,:,:]
            new_Y[new_idx,:,:,:]   = self.Y_data[old_idx,:,:,:]
        self.X_data = new_X
        self.Y_data = new_Y
        
    def reset(self, shuffle=False):
        """
        Resets the data. Used after every epoch
        """
        self.batch_num = 0
        if shuffle:
            self.shuffle()
            
def split_data(sequences, labels):
    """
    Split the data into a train and test dataset
    """
    seq_and_labels = Data(sequences, labels)
    seq_and_labels.shuffle()
    train_seq, train_labels = seq_and_labels.next_batch(NUM_TRAINING_SAMPLES)
    test_seq, test_labels = seq_and_labels.next_batch(NUM_TESTING_SAMPLE)
    return train_seq, train_labels, test_seq, test_labels