%matplotlib inline

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)


inputs_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, 28, 28, 1), name='targets')

### Encoder
conv1 = tf.layers.conv2d(inputs_, 32, (3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32
maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same')
# Now 14x14x32
conv2 = tf.layers.conv2d(maxpool1, 32, (3,3), padding='same', activation=tf.nn.relu)

# Now 14x14x32
maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same')
# Now 7x7x32
conv3 = tf.layers.conv2d(maxpool2, 16, (3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
encoded = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same')
# Now 4x4x16

### Decoder
upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
# Now 7x7x16
conv4 = tf.layers.conv2d(upsample1, 16, (3,3), padding='same', activation=tf.nn.relu)
# Now 7x7x16
upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
# Now 14x14x16
conv5 = tf.layers.conv2d(upsample2, 32, (3,3), padding='same', activation=tf.nn.relu)
# Now 14x14x32
upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
# Now 28x28x32
conv6 = tf.layers.conv2d(upsample3, 32, (3,3), padding='same', activation=tf.nn.relu)
# Now 28x28x32

logits = tf.layers.conv2d(conv6, 1, (3,3), padding='same', activation=None)
#Now 28x28x1

decoded = tf.nn.sigmoid(logits, name='decoded')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(cost)


def generate_NU():
    NU=np.random.randint(100, size=(28,28,1)) 
    #mu = FLAGS.mu
    mu = 1
    NU[NU<mu] = 0
    NU[NU>=mu] = 1

    return NU

def generate_noise():
    return np.random.laplace(loc=0.0, scale=1, size=(28,28,1)).astype(np.float32)


def generate_batch(noise, batch_size):
    #batch_NU = []
    batch_noise = []
    for i in range(batch_size):
        #batch_NU.append(NU)
        batch_noise.append(noise)
    batch_noise = np.array(batch_noise)
    
    return batch_noise


sess = tf.Session()

epochs = 1
batch_size = 200

sess.run(tf.global_variables_initializer())
      
for e in range(epochs):
    NU = generate_NU()
    noise = generate_noise()

    for offset in range(0,  mnist.train.images.shape[0], batch_size):
        end = offset + batch_size
            
        batch_x =  mnist.train.next_batch(batch_size)
        batch_x = batch_x[0].reshape((-1,28,28,1))
              
        batch_N = batch_x*NU
        
        batch_noise = generate_batch(noise, len(batch_x))
        batch_N = batch_N + batch_noise
        
        # Add clip if noise or NU is too large
        #batch_N = np.clip(batch_N, 0., 1.)

        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: batch_N, targets_: batch_x,
                                                           })

        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))
        

NU = generate_NU()
noise = generate_noise()

in_imgs = mnist.test.images[:10].reshape((-1, 28, 28, 1))

batch_noise = generate_batch(noise, 10)

noisy_imgs = in_imgs*NU + batch_noise

# Add clip if noise or NU is too large
#noisy_imgs = np.clip(noisy_imgs, 0., 1.)

reconstructed = sess.run(decoded, feed_dict={inputs_: noisy_imgs})

# Plot the examples
fig,((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10))
ax1.imshow(noisy_imgs[0].reshape((28, 28)),cmap='Greys_r')

ax2.imshow(reconstructed[0].reshape((28, 28)),cmap='Greys_r')

ax4.imshow(reconstructed[9].reshape((28, 28)),cmap='Greys_r')

ax3.imshow(noisy_imgs[9].reshape((28, 28)),cmap='Greys_r')

ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')

# show original image with label 7
plt.imshow(in_imgs[0].reshape((28,28)),cmap='Greys_r')

# show orginal image with label 9
plt.imshow(in_imgs[0].reshape((28,28)),cmap='Greys_r')