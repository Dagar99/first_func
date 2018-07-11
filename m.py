import numpy as np 
import tensorflow as tf 
character=tf.Variable(tf.random_uniform([126,128])
def word_to_vec(word):
      conv1 = tf.layers.conv2d(
      input_shape=[None,128],
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=None)
      bias=tf.Variable(tf.zeros([128]))
      f=tf.tanh(tf.add(conv1+bias))
      y=tf.layers.max_pooling2d(t,[2,2],1,padding='same')
      return y
