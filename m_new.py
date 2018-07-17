from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Flatten
character=np.random.random((128,128))
Wt=np.random.random((128,128))
bt=np.zeros(128)
Wh=np.random.random((128,128))
bh=np.zeros(128)
# I have not used model fit anywhwere as it is just a prototype
def word_to_vec(word):
    i=0
    data=np.empty((len(word),128))
    for i in range(len(word)):
        data[i]=character[ord(word[i])] 
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 padding="same",use_bias=True,bias_initializer='zeros'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Flatten())
    y=model
    model.add(Dense(128,activation='sigmoid',use_bias=True,bias_initializer='zeros'))
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.SGD(),
                  metrics=['accuracy'])
    t=model
    return t,y

def lstm(sentence):
    lis=sentence.split()
    i=0
    T=[]
    Y=[]
    for i in range(len(word)):
        t,y=word_to_vec(lis[i])
        T.append(t)
        Y.append(y)
    model=Sequential()
    model.add(Dense(128,use_bias=True,bias_initializer='zeros',activation='relu'))
    model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
    z=np.matmul(T,model)+np.matmul((1-T),Y)  
    return z
