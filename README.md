# Not Just Privacy: Improving Performance of Private Deep Learning in Mobile Cloud

Codes for Not Just Privacy: Improving Performance of Private Deep Learning in Mobile Cloud. 

### Requiremented denpendencies.

#### 1. Performance test
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN 8.0 or CPU(not recommend)
- Tensorflow-gpu 1.3.0, keras 2.0.5, python 3.6, numpy 1.14.0, scikit-learn 0.18.1

#### 2. Implementation on Android
- Linux or macOS
- JDK 1.8
- Android Studio 2.3.3
- Android SDK 7.0, Android SDK Build Tools 26.0.1, Android SDK Tools 26.1.1, Android SDK Platform Tools 26.0.1

### Notes
`local_network.py` is the network class for local neural network.

`local_weight.npy` stores the weights of the pretrained local neural network.

`svhn_train.py` is an example of noisy training, which includes the local and the cloud cloud network. This file will generate the `svhn_noisy_25.npy` which stores the weights of trained cloud network. 

`ARDEN_test.py` is an example file used to test the proposed framework. Here we use [SVHN dataset](http://ufldl.stanford.edu/housenumbers/).

`reconstruct_mnist.py` is an example for reconstructing the original data from the perturbed data, here is for [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

`TFDroid` is a demo project on Android system for testing the time overhead of `Large-Conv` neural network in mobile devices.
