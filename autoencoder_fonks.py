#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 03:43:38 2019

@author: aylin
"""


import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from PIL import Image
from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


pil_im = Image.open('a1.jpg').convert('L')
pil_im2 = Image.open('a2.jpg').convert('L')
pil_im3 = Image.open('a3.jpg').convert('L')


out1 = np.array(pil_im.resize((128,128)))
out2 = np.array(pil_im2.resize((128,128)))
out3 = np.array(pil_im3.resize((128,128)))

    
imgs = np.array([out1,out2,out3])




#hyperparameters
batch_size = 1
original_dim = 16384
latent_dim = 2
intermediate_dim = 256
nb_epoch = 5


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.)
    return z_mean + K.exp(z_log_var / 2) * epsilon

#encoder
x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

#decoder

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

#loss
def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

def autoencoder_predict(x_compared):
    vae = Model(x, x_decoded_mean)
    vae.compile(optimizer='rmsprop', loss=vae_loss)

    #vae.summary()

    x_compared = x_compared.astype('float32') / 255.
    x_compared = x_compared.reshape((len(x_compared), np.prod(x_compared.shape[1:])))


    vae.fit(x_compared, x_compared,
        shuffle=False,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_compared, x_compared),verbose=1)


    encoder = Model(x, z_mean)


    x_compared_encoded = encoder.predict(x_compared, batch_size=batch_size)

    distances = []
    img_combinations = list(combinations(range(0,len(imgs)), 2))
    for i in img_combinations:
        
        distance_x_compared = np.linalg.norm(x_compared_encoded[i[0]]-x_compared_encoded[i[1]])
        distances.append([distance_x_compared, i])

    return distances, x_compared_encoded



x_compared = np.array([imgs[1],imgs[0], imgs[2]])
distances, x_compared_encoded = autoencoder_predict(x_compared)
plt.figure(figsize=(6, 6))
plt.scatter(x_compared_encoded[:, 0], x_compared_encoded[:, 1])
plt.show()




#clusters

eps = 5


stscaler = StandardScaler().fit(x_compared_encoded)
df = stscaler.transform(x_compared_encoded)
        
dbsc = DBSCAN(eps = eps, min_samples = 2).fit(df)

clusters = dbsc.fit_predict(x_compared_encoded)

plt.scatter(x_compared_encoded[:, 0], x_compared_encoded[:, 1], c=clusters)
