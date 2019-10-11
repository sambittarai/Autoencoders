import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
import numpy as np


#Parameters of the network
depth = 32 #how much encoding to apply, compression of factor 24.5 if MNIST 28*28
length = 28*28
batch_size = 256


#Network Architecture
#create the network
inputs = Input(shape=(length,))
encoded = Dense(depth*4, activation='relu')(inputs)
encoded = Dense(depth*2, activation='relu')(encoded)
encoded = Dense(depth, activation='relu')(encoded)

decoded = Dense(depth*2, activation='relu')(encoded)
decoded = Dense(depth*4, activation='relu')(decoded)
decoded = Dense(length, activation='sigmoid')(decoded)

#map an input to its reconstruction to create model
autoencoder = Model(inputs, decoded)

#create sub network
encoder = Model(inputs, encoded)

#Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#load the data
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


#fit 
autoencoder.fit(x_train, x_train, epochs=100, batch_size=batch_size, shuffle=True, validation_data=(x_test, x_test))


#Plotting the input image and its corresponding Reconstructed image (Autoencoder's output)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(16, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


#plot latent space  clustering
x_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(10, 10))
plt.scatter(x_encoded[:, 0], x_encoded[:, 1], c=y_test, cmap='brg')
plt.colorbar()
plt.show()