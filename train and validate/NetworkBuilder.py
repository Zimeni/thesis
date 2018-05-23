import tensorflow as tf
import datetime
import numpy as np
import os


class NetworkBuilder:
    def __init__(self):
        pass

    def attach_conv_layer(self, input_layer, output_size=32, feature_size=(5, 5), strides=[1, 1, 1, 1], padding='SAME',
                          summary=False):
        with tf.name_scope("Convolution") as scope:
            #x_image = tf.reshape(input_layer, [-1, 32, 32, 1])
            #input_size = input_layer.get_shape().as_list()[-1]
            test = input_layer.get_shape()
            print("TEst: " + str(test))
            input_size = input_layer.get_shape().as_list()[-1]

            print("Input size as a LIST: " + str(input_size))

            weights = tf.Variable(tf.random_normal([feature_size[0], feature_size[1], input_size, output_size]), name='conv_weights')
            if summary:
                tf.summary.histogram(weights.name, weights)
            biases = tf.Variable(tf.random_normal([output_size]),name='conv_biases')
            #conv = tf.nn.conv2d(input_layer, weights, strides=strides, padding=padding)+biases
            conv = tf.nn.conv2d(input_layer, weights, strides=strides, padding=padding)+biases
            print("CONV: " + str(conv))
            print("CONV SHAPE: " + str(conv.get_shape().as_list()[-1]))
            return conv

    def attach_relu_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            print("RELU: ")
            input_size = input_layer.get_shape().as_list()[-1]
            print("Input size as a LIST: " + str(input_size))
            print("RELU: " + str(tf.nn.relu(input_layer)))
            return tf.nn.relu(input_layer)

    def attach_sigmoid_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            print("attach_sigmoid_layer: ")
            input_size = input_layer.get_shape().as_list()[-1]
            print("Input size as a LIST: " + str(input_size))
            print("Sigmoid: " + str(tf.nn.sigmoid(input_layer)))
            return tf.nn.sigmoid(input_layer)

    def attach_softmax_layer(self, input_layer):
        with tf.name_scope("Activation") as scope:
            print("attach_softmax_layer: ")
            input_size = input_layer.get_shape().as_list()[-1]
            print("Input size as a LIST: " + str(input_size))
            print("Sigmoid: " + str(tf.nn.softmax(input_layer, name="outputnode")))
            return tf.nn.softmax(input_layer, name="outputnode")


    def attach_pooling_layer(self, input_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME'):
        with tf.name_scope("Pooling") as scope:
            print("attach_pooling_layer: ")
            input_size = input_layer.get_shape().as_list()[-1]
            print("Input size as a LIST: " + str(input_size))
            print("Pool: " + str(tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding=padding)))
            return tf.nn.max_pool(input_layer, ksize=ksize, strides=strides, padding=padding)

    def flatten(self, input_layer):
        with tf.name_scope("Flatten") as scope:
            input_size = input_layer.get_shape().as_list()
            new_size = input_size[-1]*input_size[-2]*input_size[-3]
            print("flatten: ")
            input_size = input_layer.get_shape().as_list()[-1]
            print("Input size as a LIST: " + str(input_size))
            print("Flat: " + str(tf.reshape(input_layer, [-1, new_size])))
            return tf.reshape(input_layer, [-1, new_size])

    def attach_dense_layer(self, input_layer, size, summary=False):
        with tf.name_scope("Dense") as scope:
            input_size = input_layer.get_shape().as_list()[-1]
            weights = tf.Variable(tf.random_normal([input_size, size]), name='dense_weigh')
            if summary:
                tf.summary.histogram(weights.name, weights)
            biases = tf.Variable(tf.random_normal([size]), name='dense_biases')
            dense = tf.matmul(input_layer, weights) + biases
            print("Dense: ")
            input_size = input_layer.get_shape().as_list()[-1]
            print("Input size as a LIST: " + str(input_size))
            print("Dense: " + str(dense))
            return dense
