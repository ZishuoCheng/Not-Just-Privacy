import tensorflow as tf
import scipy.io as sio  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
import numpy as np 
from numpy import random
import random
import time
#%matplotlib inline

import os
# select which GPU to run
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from keras.utils import *
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.contrib.layers import flatten
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold

########################################
'''svhn dataset'''

nb_classes = 10

matfn = u'test_32x32.mat' 
data=sio.loadmat(matfn)

X_test = data['X']
Y_test = data['y']

temp_test = []
for i in range(X_test.shape[3]):
    temp_test.append(X_test[:,:,:,i])
X_test = np.array(temp_test)
    
temp = data['y'].tolist()
temp = np.array(temp)
temp_y=[]
for i in range(len(temp)):
    if temp[i][0]==10:
        temp_y.append([0])
    else:
        temp_y.append([temp[i][0]])

Y_test = np_utils.to_categorical(np.array(temp_y), nb_classes)

print('Original: X_test: {}, Y_test: {}'.format(X_test.shape, Y_test.shape))
# #######################################

from Class_SVHN_retrieve import SVHN
from local_network import CIFAR100

average_acc = 0
total_acc = 0

for i in range(10):
    
    tf.reset_default_graph()
    noise = np.random.laplace(loc=0.0, scale=2, size=(16,16,64)).astype(np.float32)

    NU=np.random.randint(100, size=(32,32,3)) 

    mu = 5
    NU[NU<mu] = 0
    NU[NU>=mu] = 1

    inputs_ = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='inputs')
   
    labels_ = tf.placeholder(tf.int64, shape=(None, 10), name='labels')

    sess = tf.Session()

    cifar = CIFAR100("local_weight.npy")

    cifar.build(inputs_*NU)

    temp = cifar.conv2

    codes = tf.nn.max_pool(temp, (1,2,2,1),(1,2,2,1), padding='SAME', name='max_pool') 

    codes = tf.add(codes, noise)

    svhn = SVHN("svhn_noisy.npy")

    svhn.build(codes)

    logits = svhn.fc1

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_)  
    cost = tf.reduce_mean(cross_entropy, name='cost')

    predicted = tf.nn.softmax(logits, name='predicted')

    prediction = tf.argmax(predicted,1)

    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1), name='correct_predicted')
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy') 

    batch_size = 128

    def eval_on_data(X, y):
        total_acc = 0
        #total_loss = 0
        for offset in range(0, X.shape[0], batch_size):
            end = offset + batch_size
            X_batch = X[offset:end]
            y_batch = y[offset:end]

            X_batch = X_batch
         
            pred, acc = sess.run([prediction,accuracy_op], feed_dict={inputs_:X_batch, labels_: y_batch})
           
            total_acc += (acc * X_batch.shape[0])

        return  total_acc / X.shape[0]

    test_acc = eval_on_data(X_test, Y_test)

    print("testing Accuracy =", test_acc)
    total_acc += test_acc

print("")
average_acc = total_acc/10
print("average accuracy: {:.4f}".format(average_acc))

