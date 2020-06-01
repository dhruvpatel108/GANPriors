from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def mnist_load(data_dir, dataset='train'):
    """
    return:
    1. [-1.0, 1.0] float64 images of shape (N * H * W)
    2. int labels of shape (N,)
    3. # of datas
    """
    mnist = input_data.read_data_sets(data_dir+'/', one_hot=False)
    X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    y_train = mnist.train.labels    
    X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    y_test = mnist.test.labels
    test_dir = data_dir+'/../../test_set'
    if not os.path.exists(test_dir+'/test_x.npy'):
        np.save(data_dir+'/../../test_set/test_x.npy', X_test*2-1)
        np.save(data_dir+'/../../test_set/test_y.npy', y_test)
    
    if dataset is 'train':
        assert np.shape(X_train)==(55000, 784), 'Shape of X_train is not consistent'
        img = np.reshape(X_train, [55000, 28, 28])
        lbls = y_train        
    elif dataset is 'test':
        assert np.shape(X_test)==(10000, 784), 'Shape of X_test is not consistent'
        img = np.reshape(X_test, [10000, 28, 28])
        lbls = y_test
    else:
        raise ValueError("dataset must be 'test' or 'train'")
    img = (img * 2 - 1).astype(np.float64)
    
    return img, lbls, len(lbls)

