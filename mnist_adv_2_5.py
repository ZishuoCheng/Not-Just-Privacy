
# In[1]:

import tensorflow as tf
import scipy.io as sio  
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg
import numpy as np 
import random
import time
#%matplotlib inline

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



noise_set = 2
mu_set = 5

epoches = 1
batch_size = 128


NAME = str(noise_set) + str(mu_set)

train_dir = './train_dir_' + NAME+ '/'
if not os.path.exists(train_dir): os.makedirs(train_dir)


import sys
# #
# make a copy of original stdout routed
stdout_backup = sys.stdout
# define the log file that receives your log info
log_file = open(train_dir + "mnist_adv_"+NAME+".log", "w")
# redirect print output to log file
sys.stdout = log_file



from keras.utils import *
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from tensorflow.contrib.layers import flatten
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold



# data_dict = np.load("svhn.npy").item()
# data_dict
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('noise', noise, "strength of laplacian noise")
tf.app.flags.DEFINE_float('mu', mu_set, "mu ")
tf.app.flags.DEFINE_float('epsilon', 5, "norm length for adversarial training ")
tf.app.flags.DEFINE_float('lambda_1', 0.2, " strenght of basic loss for clean examples")
tf.app.flags.DEFINE_float('lambda_2', 0.8, " strenght of another loss for noise and adversarial examples")


#####################################################################
'''svhn dataset'''

# nb_classes = 10

# matfn = u'train_32x32.mat' 
# data=sio.loadmat(matfn)

# X_train = data['X']
# Y_train = data['y']

# temp_train = []
# for i in range(X_train.shape[3]):
#     temp_train.append(X_train[:,:,:,i])
# X_train = np.array(temp_train)
    
# temp = data['y'].tolist()
# temp = np.array(temp)
# temp_y=[]
# for i in range(len(temp)):
#     if temp[i][0]==10:
#         temp_y.append([0])
#     else:
#         temp_y.append([temp[i][0]])

# Y_train = np_utils.to_categorical(np.array(temp_y), nb_classes)

# matfn = u'test_32x32.mat' 
# data=sio.loadmat(matfn)

# X_test = data['X']
# Y_test = data['y']

# temp_test = []
# for i in range(X_test.shape[3]):
#     temp_test.append(X_test[:,:,:,i])
# X_test = np.array(temp_test)
    
# temp = data['y'].tolist()
# temp = np.array(temp)
# temp_y=[]
# for i in range(len(temp)):
#     if temp[i][0]==10:
#         temp_y.append([0])
#     else:
#         temp_y.append([temp[i][0]])

# Y_test = np_utils.to_categorical(np.array(temp_y), nb_classes)

# print('Original: X_train: {}, Y_train: {}'.format(X_train.shape, Y_train.shape))
# print('Original: X_test: {}, Y_test: {}'.format(X_test.shape, Y_test.shape))


# # In[3]:

# from sklearn.model_selection import train_test_split
# X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify = Y_train)
# print('X_train: {}, Y_train: {}'.format(X_train.shape, Y_train.shape))
# print('X_valid: {}, Y_valid: {}'.format(X_valid.shape, Y_valid.shape))
# print('X_test: {}, Y_test: {}'.format(X_test.shape, Y_test.shape))
# ##############################################################################


# # #######################################
'''mnist dataset'''
from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

nb_classes = 10

Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)


x_train = []
x_test = []

for i in range(len(X_train)):
    temp = np.stack((X_train[i],)*3, axis=2)
    x_train.append(temp)

for i in range(len(X_test)):
    temp = np.stack((X_test[i],)*3, axis=2)
    x_test.append(temp)

X_train, X_test = np.array(x_train), np.array(x_test)


print("Mnist Dataset")
print('Original: X_train: {}, Y_train: {}'.format(X_train.shape, Y_train.shape))
print('Original: X_test: {}, Y_test: {}'.format(X_test.shape, Y_test.shape))
print("")

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify = Y_train)
print('X_train: {}, Y_train: {}'.format(X_train.shape, Y_train.shape))
print('X_valid: {}, Y_valid: {}'.format(X_valid.shape, Y_valid.shape))
print('X_test: {}, Y_test: {}'.format(X_test.shape, Y_test.shape))
# #######################################



# In[3]:

# In[4]:

# X_train, X_valid, X_test = X_train[:18], X_valid[:18], X_test[:18]
# Y_train, Y_valid, Y_test = Y_train[:18], Y_valid[:18], Y_test[:18]


# In[4]:

from Pretrained_CIFAR100 import CIFAR100

# cifar=CIFAR100("cifar100.npy")
    
def local_extract(inputs_):
    
    cifar.build(inputs_)
    
    codes = cifar.conv2
    
    codes = tf.nn.max_pool(codes, (1,2,2,1),(1,2,2,1), padding='SAME', name='pool0_new') 

    codes = tf.stop_gradient(codes)
    
    return codes



def basic_model(inputs_, labels_, is_training=True):
    
   # print('debug codes:', codes.shape)

     # conv1_1
    with tf.name_scope('conv1_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inputs_, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1 = lrelu(out, name=scope)
       
        
    with tf.name_scope('conv2_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 96, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2 = lrelu(out, name=scope)
       
        
    with tf.name_scope('conv3_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 96, 96], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3 = lrelu(out, name=scope)
       

    maxpool1 = tf.nn.max_pool(conv3, (1,2,2,1),(1,2,2,1), padding='SAME', name='pool1_new')   
    
    drop1 = Dropout(0.5, name='drop1_new')(maxpool1)
    
    with tf.name_scope('conv4_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 96, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(drop1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4 = lrelu(out, name=scope)
      
        
    with tf.name_scope('conv5_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5 = lrelu(out, name=scope)
      
    with tf.name_scope('conv6_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv6 = lrelu(out, name=scope)
       
        
    maxpool2 = tf.nn.max_pool(conv6, (1,2,2,1),(1,2,2,1), padding='SAME', name='pool2_new')   
    
    drop2 = Dropout(0.5, name='drop2_new')(maxpool2)
     
    
    with tf.name_scope('conv7_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(drop2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv7 = lrelu(out, name=scope)
      
    
    with tf.name_scope('conv8_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv8 = lrelu(out, name=scope)
      
    
    with tf.name_scope('conv9_new') as scope:
        kernel = tf.Variable(tf.truncated_normal([1, 1, 192, 192], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv9 = lrelu(out, name=scope)
       
    
    h = tf.reduce_mean(conv9, reduction_indices=[1, 2],name='ave_pool_new')
    
    #print('debug h: ', h.shape)
    
    with tf.name_scope('fc1_new') as scope:
        #shape = int(np.prod(h.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([192, 10],
                                                     dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[10], dtype=tf.float32),
                             trainable=True, name='biases')
        #pool5_flat = tf.reshape(h, [-1, shape])
        logits = tf.nn.bias_add(tf.matmul(h, fc1w), fc1b)
        #fc1 = lrelu(fc1l,name=scope)
        #self.parameters += [fc1w, fc1b]
        #print('debug fc1: ', fc1.shape)   

    #with tf.name_scope('loss') as scope:

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_)  
    cost = tf.reduce_mean(cross_entropy, name='cost')

    if is_training == True:    
        # Operations for validation/test accuracy
        predicted = tf.nn.softmax(logits, name='predicted')
        correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1), name='correct_predicted')
        accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32),name='accuracy') 

        return cost, logits, accuracy_op
    else:
        return cost



def lrelu(x,name="default"):
    return tf.maximum(x, 0.1 * x)
    


# def adv_model(X, Y, NU, noise):
    
    
#     # add nullification
#     X = X*NU
#     codes = local_extract(X)
    
#     # add laplacian noise
#     codes = tf.add(codes, noise)

#     loss, logit, accuracy_operation = basic_model(codes, Y, is_training=True)

#     gradx = tf.stop_gradient(tf.gradients(loss, [codes], aggregation_method=2))[0]
#     gradx = gradx / (1e-12 + tf.reduce_max(tf.abs(gradx), reduction_indices=[1,2], keep_dims=True))
#     gradx = gradx / tf.sqrt(1e-6 + tf.reduce_sum(gradx**2, reduction_indices=[1,2], keep_dims=True))
#     radv = FLAGS.epsilon * gradx
#     adv_loss = basic_model(codes+radv, Y, is_training=False)
    
#     another_loss = loss + adv_loss

#     cost = another_loss

#     return cost, accuracy_operation # adv_Loss for debug


# In[9]:

def adv_model(X, Y, NU, noise):
    
    
    codes = local_extract(X)
    basic_loss = basic_model(codes, Y, is_training=False)
    
    # add nullification
    X = X*NU
    codes = local_extract(X)
    
    # add laplacian noise
    codes = tf.add(codes, noise)

    loss, logit, accuracy_operation = basic_model(codes, Y, is_training=True)

    gradx = tf.stop_gradient(tf.gradients(loss, [codes], aggregation_method=2))[0]
    gradx = gradx / (1e-12 + tf.reduce_max(tf.abs(gradx), reduction_indices=[1,2], keep_dims=True))
    gradx = gradx / tf.sqrt(1e-6 + tf.reduce_sum(gradx**2, reduction_indices=[1,2], keep_dims=True))
    radv = FLAGS.epsilon * gradx
    adv_loss = basic_model(codes+radv, Y, is_training=False)
    another_loss = loss + adv_loss

    cost = FLAGS.lambda_1*basic_loss + FLAGS.lambda_2*another_loss

    return cost, accuracy_operation # adv_Loss for debug


# NU=np.random.randint(100, size=(32,32,3)) 
# mu = FLAGS.mu
# NU[NU<mu] = 0
# NU[NU>=mu] = 1

# noise = np.random.laplace(loc=0.0, scale=FLAGS.noise, size=(16,16,64)).astype(np.float32)


# In[14]:


# In[19]:

def generate_NU():
    NU=np.random.randint(100, size=(32,32,3)) 
    mu = FLAGS.mu
    NU[NU<mu] = 0
    NU[NU>=mu] = 1

    return NU

def generate_noise():
    return np.random.laplace(loc=0.0, scale=FLAGS.noise, size=(16,16,64)).astype(np.float32)


def generate_batch(NU, noise, batch_size):
    batch_NU = []
    batch_noise = []
    for i in range(batch_size):
        batch_NU.append(NU)
        batch_noise.append(noise)
    batch_NU, batch_noise = np.array(batch_NU), np.array(batch_noise)
    
    return batch_NU, batch_noise

# In[11]:

# In[15]:

def eval_on_data(X, y):    
    # generate Nullification and Laplacian noise
    NU = generate_NU()
    noise = generate_noise()
    
    total_acc = 0
    #total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]
        
        batch_NU, batch_noise = generate_batch(NU, noise, len(X_batch))
        
        acc = sess.run(accuracy_op, feed_dict={inputs_: X_batch, labels_: y_batch, 
                                              NU_: batch_NU, noise_: batch_noise, K.learning_phase():0})
        # print('valid: , acc: ', loss, acc)
        #total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])
        #print('debug: ',X_batch.shape[0])

    return  total_acc / X.shape[0]


# In[12]:

inputs_ = tf.placeholder(tf.float32, shape=(None, 28, 28, 3), name='inputs')
resized_ = tf.image.resize_images(inputs_, (32, 32))
labels_ = tf.placeholder(tf.int64, shape=(None, 10), name='labels')
NU_ = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='nullification')
noise_ = tf.placeholder(tf.float32, shape=(None, 16, 16, 64), name='noise')
#total_loss, accuracy_op = adv_model(inputs_, labels_)

# codes = local_extract(inputs_)
# total_loss, _, accuracy_operation = basic_model(codes, labels_)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_)  
# cost = tf.reduce_mean(cross_entropy)

# # Operations for validation/test accuracy
# predicted = tf.nn.softmax(logits)
# correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
# accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  


# In[20]:

# cifar=CIFAR100("cifar100.npy")
# cifar.build(inputs_)

# codes = local_extract(inputs_)
# basic_loss = basic_model(codes, labels_, is_training=False)

# another_loss, accuracy_op = adv_model(inputs_, labels_, NU_, noise_)

# cost = FLAGS.lambda_1*basic_loss + FLAGS.lambda_2*another_loss


# In[21]:


# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess:
    
    K.set_session(sess)
    
    # # basic model    
    # cifar=CIFAR100("cifar100.npy")
    # cifar.build(inputs_)
    # cost, _, accuracy_op = basic_model(inputs_, labels_)

    # advmodel
#     cifar=CIFAR100("cifar100.npy")
#     cifar.build(inputs_)
    
#     codes = local_extract(inputs_)
#     basic_loss = basic_model(codes, labels_, is_training=Flase)
    
#     another_loss, accuracy_op = adv_model(inputs_, labels_, mu_, noise_)
    
#     cost = FLAGS.lambda_1*basic_loss + FLAGS.Lambda_2*another_loss

#     optimizer = tf.train.AdamOptimizer(0.0015).minimize(cost)  # training optimizer
    
    cifar=CIFAR100("cifar100.npy")
    #cifar.build(inputs_)
    cost, accuracy_op = adv_model(resized_, labels_, NU_, noise_)
    optimizer = tf.train.AdamOptimizer(0.0015).minimize(cost)  # training optimizer


    sess.run(tf.global_variables_initializer())
   
    saver = tf.train.Saver()
    for i in range(epoches):
        print('------------>Epoch:    ', i)
        batch_accuracy = 0

        t0 = time.time()
        ii = 0
        
        # generate Nullification and Laplacian noise
        NU = generate_NU()
        noise = generate_noise()
        
        
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            
            
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            
            batch_NU, batch_noise = generate_batch(NU, noise, len(batch_x))
            
            #codes =sess.run(middle, feed_dict={inputs_: batch_x, labels_:batch_y})
            
            #batch_x = sess.run(middle, feed_dict={inputs_:batch_x, labels_:batch_y})
            batch_loss, batch_acc, _ = sess.run([cost, accuracy_op, optimizer],
                                                feed_dict={inputs_: batch_x, labels_: batch_y,
                                                NU_: batch_NU, noise_: batch_noise, K.learning_phase():1})
            batch_accuracy += (batch_acc * len(batch_x))
            
            ii = ii + 1
            if ii%100==0:
                print("Epoch: {}/{}...".format(i + 1, ii),
                    "Training batch loss: {:.4f}".format(batch_loss),
                     "Training batch acc: {:.4f}".format(batch_acc))


        print('')
        #print('Training loss: ', batch_loss)
        print("Epoch: {}/{}...".format(i + 1, epoches),
              "Training accuracy: {:.4f}".format(batch_accuracy / len(X_train)))
        print("Time: %.3f seconds" % (time.time() - t0))
        print('')

        val_acc = eval_on_data(X_valid, Y_valid)
        print("Epoch", i + 1)
        
        #print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")
    

    # test_acc = eval_on_data(X_test, Y_test)
   
    # #print("Testing Loss =", val_loss)
    # print("Testing Accuracy =", test_acc)
    # print("")    
    
    dict_list = {}
    for v in tf.trainable_variables(): 
        #print(v.name) # 
        ww = sess.run(v) #
        dict_list[v.name] = ww
        #print(ww.shape)
    np.save(train_dir + "mnist_adv_"+NAME+".npy", dict_list)
   
    #Save the variables to disk.
    save_path = saver.save(sess, train_dir + "mnist_adv_" + NAME + ".ckpt")
    print("Model saved in file: %s" % save_path)



#saver = tf.train.Saver()
with tf.Session() as sess:
    # Restore variables from disk.
    #saver.restore(sess, "./checkpoints/basic_model.ckpt")
    saver.restore(sess, tf.train.latest_checkpoint(train_dir + "mnist_adv_"+NAME + ".ckpt"))
    print("Model restored.")
    # Check the values of the variables
    test_acc = eval_on_data(X_test, Y_test)
   
    #print("Testing Loss =", val_loss)
    print("Testing Accuracy =", test_acc)
    print("")
          
log_file.close()  



