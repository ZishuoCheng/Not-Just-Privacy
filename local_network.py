import tensorflow as tf
import numpy as np


class  CIFAR100:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding = 'latin1').item()
        print("npy file loaded")
    
    def lrelu(self, x,name="default"):
        return tf.maximum(x, 0.1 * x)
   
    def build(self, inputs):
        
        with tf.name_scope('conv1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = self.lrelu(out, name=scope)
          
        
        with tf.name_scope('conv2') as scope:
             
            kernel =  tf.constant(self.data_dict['conv2/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv2/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = self.lrelu(out, name=scope)
            
        with tf.name_scope('conv3') as scope:
             
            kernel =  tf.constant(self.data_dict['conv3/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv3/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = self.lrelu(out, name=scope)
        
        self.maxpool1 = tf.nn.max_pool(self.conv3, (1,2,2,1),(1,2,2,1), padding='SAME', name='pool1')   
    
        
