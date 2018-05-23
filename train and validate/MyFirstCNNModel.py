import tensorflow as tf
from NetworkBuilder import NetworkBuilder
from DataSetGenerator import DataSetGenerator, seperateData
import datetime
import numpy as np
import os
#from tensorflow.python.tools import freeze_graph


with tf.name_scope("Input") as scope:
    #input_img = tf.placeholder(dtype='float', shape=(None, 128, 128, 3), name="input")
    input_img = tf.placeholder(dtype='float', shape=(None, 252, 252, 3), name="input")


with tf.name_scope("Output") as scope:
    #target_labels = tf.placeholder(dtype='float', shape=[None, 3], name="output")
    target_labels = tf.placeholder(dtype='float', shape=[None, 8], name="output")

with tf.name_scope("Keep_prob_input") as scope:
    keep_prob = tf.placeholder(dtype='float',name='keep_prob')

nb = NetworkBuilder()

with tf.name_scope("ModelV2") as scope:
    model = input_img
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 32, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 64, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_conv_layer(model, 128, summary=True)
    model = nb.attach_relu_layer(model)
    model = nb.attach_pooling_layer(model)

    model = nb.flatten(model)
    model = nb.attach_dense_layer(model, 200, summary=True)
    model = nb.attach_sigmoid_layer(model)
    model = nb.attach_dense_layer(model, 32, summary=True)
    model = nb.attach_sigmoid_layer(model)
    #model = nb.attach_dense_layer(model, 3)
    model = nb.attach_dense_layer(model, 8)
    prediction = nb.attach_softmax_layer(model)
    output = tf.identity(prediction, 'outt')
    print("CNN END: ")


with tf.name_scope("Optimization") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=target_labels)
    cost = tf.reduce_mean(cost)
    tf.summary.scalar("cost", cost)
    optimizer = tf.train.AdamOptimizer().minimize(cost, global_step=global_step)
    print("Optimization: ")

with tf.name_scope('accuracy') as scope:
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("accuracy: ")

dgTrain = DataSetGenerator("train/")
dgValidate = DataSetGenerator("val/")

epochs = 10
batchSize = 10

saver = tf.train.Saver()
print("saver = tf.train.Saver(): ")
model_save_path="./thesis/"
model_name='thesis'

with tf.device("/gpu:0"):
    with tf.Session() as sess:
        summaryMerged = tf.summary.merge_all()

        filename = "./summary_log/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
        # setting global steps
        tf.global_variables_initializer().run()
        sess.run(tf.local_variables_initializer())

        if os.path.exists(model_save_path+'checkpoint'):
            # saver = tf.train.import_meta_graph('./saved '+modelName+'/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint(model_save_path))
            print("saver.restore(sess, tf.train.latest_checkpoint(model_save_path)): ")

        writer = tf.summary.FileWriter(filename, sess.graph)

        for epoch in range(epochs):
            #batches = dgTrain.get_mini_batches(batchSize,(128,128), allchannel=True)
            batches = dgTrain.get_mini_batches(batchSize,(252,252), allchannel=True)

            for imgs ,labels in batches:
                print("IMAGES SHAPE: " + str(imgs.size))
                imgs=np.divide(imgs, 255)

                optimizer.run(feed_dict={input_img: imgs, target_labels: labels})

                if (epoch+1) % 1 == 0:
                    #validateBatch = dgValidate.get_mini_batches(batchSize,(128,128), allchannel=True)
                    validateBatch = dgValidate.get_mini_batches(batchSize,(252,252), allchannel=True)
                    for imgs, labels in validateBatch:
                        train_accuracy = accuracy.eval(feed_dict={
                          input_img: imgs, target_labels: labels})
                        print("step %d, training accuracy %g"%(epoch+1, train_accuracy))

        print("Saving the model")
        saver.save(sess, model_save_path + model_name + '.chkp')
