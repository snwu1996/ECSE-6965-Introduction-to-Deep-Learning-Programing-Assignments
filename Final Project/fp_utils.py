import numpy as np
import tensorflow as tf

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
        X_batch = self.X_data[self.batch_num*batch_size:(self.batch_num+1)*batch_size]
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
        X_batch = self.X_data[rand_nums]
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
            new_X[new_idx] = self.X_data[old_idx]
            new_Y[new_idx]   = self.Y_data[old_idx]
        self.X_data = new_X
        self.Y_data = new_Y
        
    def reset(self, shuffle=False):
        """
        Resets the data. Used after every epoch
        """
        self.batch_num = 0
        if shuffle:
            self.shuffle()

##############################################################################
            
def split_data(sequences, labels, percent_train=.7, shuffle=True):
    """
    Split the data into a train and test dataset
    """
    seq_and_labels = Data(sequences, labels)
    if shuffle:
        seq_and_labels.shuffle()
    num_training_samples = int(sequences.shape[0]*percent_train)
    num_testing_sample = sequences.shape[0]-num_training_samples
    train_seq, train_labels = seq_and_labels.next_batch(num_training_samples)
    test_seq, test_labels = seq_and_labels.next_batch(num_testing_sample)
    return train_seq, train_labels, test_seq, test_labels

def train_test_split(sequences, labels):
    seqs_per_action_index = np.bincount(labels)
    seqs_per_action_index = np.cumsum(seqs_per_action_index)
    # These indexes are based on a 70 30 split and then manually corrected 
    # for to make sure no single video overlaps between train test
    train_test_split_indexs = [426,480,510,455,656,545,453,533,390,327,400]
    seqs_per_class = np.split(sequences, seqs_per_action_index)
    labels_per_class = np.split(labels, seqs_per_action_index)

    train_seqs = []
    train_labels = []
    test_seqs = []
    test_labels = []

    for seq_per_class, label_per_class, train_test_split_index in zip(seqs_per_class,
                                                                      labels_per_class,
                                                                      train_test_split_indexs):
        train_sequence = seq_per_class[:train_test_split_index]
        train_label = label_per_class[:train_test_split_index]
        test_sequence = seq_per_class[train_test_split_index:]
        test_label = label_per_class[train_test_split_index:]

        train_seqs.append(train_sequence)
        train_labels.append(train_label)
        test_seqs.append(test_sequence)
        test_labels.append(test_label)

    del seqs_per_class, labels_per_class

    train_seqs = np.concatenate(train_seqs, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    test_seqs = np.concatenate(test_seqs, axis=0)
    test_labels = np.concatenate(test_labels, axis=0) 
    
    return train_seqs, train_labels, test_seqs, test_labels

def get_testing_loss_and_error(sess, test_data, test_acc_ops, test_batch_size):
    num_test_batches = int(test_data.Y_data.shape[0]/test_batch_size)+1
    test_loss = 0.
    test_acc = 0.
    test_data.reset(shuffle=False)
    for test_batch in range(num_test_batches):
        X_batch_test, Y_batch_test = test_data.next_batch(test_batch_size)
        test_loss_batch, test_acc_batch = sess.run([test_acc_ops['loss'], test_acc_ops['accuracy_op']],
                                                   feed_dict={test_acc_ops['X']: X_batch_test, 
                                                              test_acc_ops['Y']: Y_batch_test})
        test_loss += test_loss_batch
        test_acc += test_acc_batch
    test_loss /= num_test_batches
    test_acc /= num_test_batches
    return test_loss, test_acc

##############################################################################

l2_reg = tf.contrib.layers.l2_regularizer(scale=1.0)
relu = tf.nn.relu
xavier_init = tf.contrib.layers.xavier_initializer()
zero_init = tf.zeros_initializer()

def convolutional_layer(inputs, layer_param):
    conv_layer = tf.layers.conv3d(inputs=inputs,
                                  filters=layer_param['filters'],
                                  kernel_size=layer_param['kernel_size'],
                                  padding=layer_param['padding'],
                                  activation=relu,
                                  kernel_initializer=xavier_init,
                                  bias_initializer=zero_init,
                                  kernel_regularizer=l2_reg,
                                  bias_regularizer=l2_reg,
                                  name=layer_param['name'])
    return conv_layer

def pooling_layer(inputs, layer_param):
    pool_layers = tf.layers.max_pooling3d(inputs=inputs,
                                          pool_size=(1,2,2),
                                          strides=(1,2,2),
                                          name=layer_param['name'])
    return pool_layers

def fc_layer(inputs, layer_param, training):
    fc_layer = tf.layers.dense(inputs=inputs,
                               units=layer_param['units'],
                               activation=relu,
                               kernel_initializer=xavier_init,
                               bias_initializer=zero_init,
                               kernel_regularizer=l2_reg,
                               bias_regularizer=l2_reg,
                               name=layer_param['name'])
#     fc_layer = tf.layers.batch_normalization(inputs=fc_layer,
#                                              momentum=layer_param['bn_momentum'],
#                                              training=training,
#                                              name='{}_bn'.format(layer_param['name']))
    fc_layer = tf.layers.dropout(inputs=fc_layer,
                                 rate=layer_param['drop_rate'],
                                 training=training,
                                 name='{}_dropout'.format(layer_param['name']))
    return fc_layer


def flat_layer(inputs, layer_param):
    flat_layer = tf.reshape(inputs, 
                            [-1, inputs.shape[1], np.prod(inputs.shape[2:])], 
                            name=layer_param['name'])
    return flat_layer

def lstm_cells(inputs, layer_param):
    lstm_cells = tf.nn.rnn_cell.LSTMCell(num_units=layer_param['units'], 
                                         activation=tf.nn.tanh,
                                         initializer=xavier_init)
    output, final_state = tf.nn.dynamic_rnn(lstm_cells, 
                                            inputs, 
                                            dtype=tf.float32)
#     output = tf.reduce_sum(output,axis=1)
    return final_state[1]

def multi_lstm_cells(inputs, layer_param):
    multi_lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=layer_param['units'], 
                                                activation=tf.nn.tanh,
                                                initializer=xavier_init)\
                                                for _ in range(layer_param['num_layers'])]
    multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(multi_lstm_cells)
    output, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, 
                                            inputs, 
                                            dtype=tf.float32)
#     output = tf.reduce_sum(output,axis=1)
    return final_state[-1][1]

def logits(inputs, layer_param):
    pred_op = tf.layers.dense(inputs, 
                              units=layer_param['num_outputs'],
                              name=layer_param['name'])
    return pred_op

def model_factory(model_params, input_layer, training):
    layers = [input_layer]

    for layer_param in model_params:
        if layer_param['layer_type'] == 'conv':
            layers.append(convolutional_layer(layers[-1], layer_param))
        elif layer_param['layer_type'] == 'pool':
            layers.append(pooling_layer(layers[-1], layer_param))
        elif layer_param['layer_type'] == 'flat':
            layers.append(flat_layer(layers[-1], layer_param))
        elif layer_param['layer_type'] == 'fc':
            layers.append(fc_layer(layers[-1], layer_param, training))
        elif layer_param['layer_type'] == 'lstm':
            layers.append(lstm_cells(layers[-1], layer_param))
        elif layer_param['layer_type'] == 'multi_lstm':
            layers.append(multi_lstm_cells(layers[-1], layer_param))
        elif layer_param['layer_type'] == 'logits':
            layers.append(logits(layers[-1], layer_param))
        else:
            raise Exception('{} is not a valid layer type'.format(layer_param['layer_type']))

    return layers