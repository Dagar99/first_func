import numpy as np 
import tensorflow as tf 
character=tf.Variable(tf.random_uniform([126,128]))
Wt=tf.Variable(tf.random_normal([128,128]))
bt=tf.Variable(tf.zeros([128]))
Wh=tf.Variable(tf.random_normal([128,128]))
bh=tf.Variable(tf.zeros([128]))

def word_to_vec(word):
      i=0
      data=[len(word),128]
      for i in range(len(word)):
           data[i]=character[ord(word[i])] 
      conv1 = tf.layers.conv2d(data,
      input_shape=[None,128],
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=None)
      bias=tf.Variable(tf.zeros([128]))
      f=tf.tanh(tf.add(conv1+bias))
      y=tf.layers.max_pooling2d(t,[2,2],1,padding='same')
      return y

def t_calc(y):
      a=tf.matmul(Wt,y)
      b=tf.add(a,bt)
      t=tf.sigmoid(b)
      return t
def lstm(sentence):
      lis=sentence.split()
      i=0
      for i in range(len(lis)):
            word=word_to_vec(lis[i])
            t=t_calc(word)
            g=tf.nn.relu(tf.add(tf.matmul(Wh,y)+bh))
            z+=tf.add(tf.multiply(t,g)+tf.multiply((1-t),word))
      return z
