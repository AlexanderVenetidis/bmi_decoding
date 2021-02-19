"""
Customise data generator for training model in Keras
"""

import numpy as np
from tensorflow.keras.utils import Sequence

class data_generator(Sequence):
    'Generates data for Keras'
    def __init__(self, X, Y, batch_size, shuffle):
        'Initialization'
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate data
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X_batch = np.empty((self.batch_size,self.X.shape[1],self.X.shape[2]))
        Y_batch = np.empty((self.batch_size,self.Y.shape[1]))
        for i, idx in enumerate(indexes):        
            X_batch[i,:] = self.X[idx,:]
            Y_batch[i,:] = self.Y[idx,:]
        
        return X_batch, Y_batch