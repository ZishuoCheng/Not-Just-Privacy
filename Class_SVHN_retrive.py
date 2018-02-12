import tensorflow as tf
import numpy as np

class SVHN:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding = 'latin1').item()
        print("npy file loaded")
    
    def lrelu(self, x, name="default"):
        return tf.maximum(x, 0.1 * x)
    
    def build(self, inputs):
                 
        with tf.name_scope('conv1_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv1_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv1_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv1 = self.lrelu(out, name=scope)
            
        
        with tf.name_scope('conv2_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv2_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv2_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv2 = self.lrelu(out, name=scope)
            
        with tf.name_scope('conv3_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv3_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv3_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv3 = self.lrelu(out, name=scope)
        
        self.maxpool1 = tf.nn.max_pool(self.conv3, (1,2,2,1),(1,2,2,1), padding='SAME', name='pool1_new')   
    

        with tf.name_scope('conv4_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv4_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.maxpool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv4_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv4 = self.lrelu(out, name=scope)
        
        with tf.name_scope('conv5_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv5_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv5_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv5 = self.lrelu(out, name=scope)
        
        with tf.name_scope('conv6_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv6_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv5, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv6_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv6 = self.lrelu(out, name=scope)
            
        self.maxpool2 = tf.nn.max_pool(self.conv6, (1,2,2,1),(1,2,2,1), padding='SAME', name='pool2_new')   
        
        with tf.name_scope('conv7_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv7_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.maxpool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv7_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv7 = self.lrelu(out, name=scope)
        
        with tf.name_scope('conv8_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv8_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv7, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv8_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv8 = self.lrelu(out, name=scope)
            
        with tf.name_scope('conv9_new_1') as scope:
             
            kernel =  tf.constant(self.data_dict['conv9_new_1/weights:0'], name="weights")
            
            conv = tf.nn.conv2d(self.conv8, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.constant(self.data_dict['conv9_new_1/biases:0'], name="biases")
            out = tf.nn.bias_add(conv, biases)
            self.conv9 = self.lrelu(out, name=scope)
            
        self.h = tf.reduce_mean(self.conv9, reduction_indices=[1, 2],name='ave_pool_new')
        
        with tf.name_scope('fc1_new_1') as scope:
          
            fc1w = tf.constant(self.data_dict['fc1_new_1/weights:0'], name="weights")
            fc1b = tf.constant(self.data_dict['fc1_new_1/biases:0'], name="biases")
            self.fc1 = tf.nn.bias_add(tf.matmul(self.h, fc1w), fc1b)



