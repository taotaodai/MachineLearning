# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


#a = tf.placeholder(tf.int32,shape=[2,2])
#b = tf.placeholder(tf.int32,shape=[2,2])
#
#c = tf.matmul(a,b)
#
#a_1 = np.array([[1,2],[3,4]])
#b_1 = np.array([[4,3],[2,1]])
#
#with tf.Session() as sess:
#    print(sess.run(c,feed_dict={a:a_1,b:b_1}))

from tensorflow.examples.tutorials.mnist import input_data
import os

data_path = os.path.join('.','temp','data')

#mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
mnist = input_data.read_data_sets(data_path,one_hot=True)

#print(mnist.train.images.shape,mnist.train.labels.shape)
#print(mnist.test.images.shape,mnist.test.labels.shape)
#print(mnist.validation.images.shape,mnist.validation.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

"定义Softmax Regression算法"
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder(tf.float32,[None,10])

"""定义损失函数"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

"定义优化算法并开始训练"
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
#    print(batch_xs.shape)

