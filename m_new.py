from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D,Flatten,LSTM
from keras.layers.wrappers import Bidirectional
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
    model.add(Dense(128,use_bias=True,bias_initializer='zeros',activation='relu')
    model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
    z=np.matmul(T,model)+np.matmul((1-T),Y)  
    return z
 def intra_attention():
             
              lstm1, state_h, state_c=(Bidirectional(LSTM(128,activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                   recurrent_regularizerropout,dropout=0.4, recurrent_dropout=0.3)))
              model=Model(inputs=lstm(y), outputs=[lstm1, state_h, state_c])
              model.compile(optimizer='rmsprop',loss='categorical crossentropy',metrics='accuracy')
              
              output=model.predict(lstm(y))
              Ww=np.random.random(128,128)
              bw=np.zeros(128)
              ut=tanh(np.dot(Ww,output[1])+bw)
              alpha_t=softmax(ut)
              s=np.multiply(alpha_t,output[1])
              return s
              
