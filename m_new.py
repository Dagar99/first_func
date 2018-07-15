from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
character=tf.Variable(tf.random_uniform([126,128]))
Wt=np.random.random(128,128)
bt=np.zeros(128)
Wh=np.random.random(128,128)
bh=np.zeros(128)
'''following func calculates the value of t but not the intermediate vector y'''
def word_to_vec(word):
    i=0
    data=[len(word),128]
    for i in range(len(word)):
        data[i]=character[ord(word[i])] 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape,padding="same",use_bias=True,bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten())
    model.add(Dense(128,activation='sigmoid',use_bias=True,bias_initializer='zeros'))
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    return model

