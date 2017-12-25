"""
Arian Niaki
Dec 2017, UMass Amherst
This is the main class for running the supervised learning algorithm for the HTML part
"""


import glob
import os
from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf
import datasets_html_cnn
import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import timedelta

def precision_recall_blockpage(y_true, y_pred):
    num_correctly_detected_blockpages = 0
    detected_blockpage = 0
    true_blockpage = 0
    for i in range(len(y_true)):
        if(y_pred[i]=='block' and y_true[i]=='block'):
            num_correctly_detected_blockpages += 1

        if(y_pred[i]=='block'):
            detected_blockpage += 1
        if(y_true[i]=='block'):
            true_blockpage += 1
    print("num correctly detected blockpages ", num_correctly_detected_blockpages)
    print("detected blockpages ", detected_blockpage)
    print("true blockpages ", true_blockpage)
    try:
        precision = (num_correctly_detected_blockpages+0.0) / (detected_blockpage+0.0)
        recall = (num_correctly_detected_blockpages+0.0) / (true_blockpage+0.0)

        print("precision is : %s  recall is : %s " %(str(precision),str(recall)))
    except Exception as exp:
        print("exception ",str(exp))

def precision_recall_servererr(y_true, y_pred):
    num_correctly_detected_servererrs = 0
    detected_servererr = 0
    true_servererr = 0
    for i in range(len(y_true)):
        if (y_pred[i] == 'servererr' and y_true[i] == 'servererr'):
            num_correctly_detected_servererrs += 1

        if (y_pred[i] == 'servererr'):
            detected_servererr += 1
        if (y_true[i] == 'servererr'):
            true_servererr += 1
    print("num correctly detected servererrs ", num_correctly_detected_servererrs)
    print("detected servererrs ", detected_servererr)
    print("true servererrs ", true_servererr)
    try:
        precision = (num_correctly_detected_servererrs + 0.0) / (detected_servererr + 0.0)
        recall = (num_correctly_detected_servererrs + 0.0) / (true_servererr + 0.0)

        print("precision is : %s  recall is : %s For Server Error" % (str(precision), str(recall)))
    except Exception as exp:
        print("exception ", str(exp))

def precision_recall_connectionerr(y_true, y_pred):
    num_correctly_detected_connectionerrs = 0
    detected_connectionerr = 0
    true_connectionerr = 0
    for i in range(len(y_true)):
        if (y_pred[i] == 'connectionerr' and y_true[i] == 'connectionerr'):
            num_correctly_detected_connectionerrs += 1

        if (y_pred[i] == 'connectionerr'):
            detected_connectionerr += 1
        if (y_true[i] == 'connectionerr'):
            true_connectionerr += 1
    print("num correctly detected connectionerrs ", num_correctly_detected_connectionerrs)
    print("detected connectionerrs ", detected_connectionerr)
    print("true connectionerrs ", true_connectionerr)
    try:
        precision = (num_correctly_detected_connectionerrs + 0.0) / (detected_connectionerr + 0.0)
        recall = (num_correctly_detected_connectionerrs + 0.0) / (true_connectionerr + 0.0)

        print("precision is : %s  recall is : %s For Connection Error" % (str(precision), str(recall)))
    except Exception as exp:
        print("exception ", str(exp))








# train_path = '/home/arian/Desktop/ML_Tensor/data/http/train/'
# test_path = '/home/arian/Desktop/ML_Tensor/data/http/test/'
train_path = 'data/train/'
test_path = 'data/test/'
# class info
classes = ['block', 'ok','servererr','connectionerr']
num_classes = len(classes)


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
# Convolutional Layer 1.
filter_size1 = 3
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32



# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.


# batch size
batch_size = 89

# validation split
validation_size = .2

num_channels = 300
print("going to read dataset")
start_time = time.time()
data = datasets_html_cnn.read_train_sets(train_path, classes, validation_size=validation_size)
import gc
gc.collect()
# Ending time.
end_time = time.time()

# Difference between start and end-times.
time_dif = end_time - start_time

# Print the time-usage.
print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


data_test = datasets_html_cnn.read_test_set_labeled(test_path, classes)



print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data_test.test.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))


# print("Loaded GloVe Matrix to Embeddings")
#
# HTML_DIR = '/home/arian/Desktop/ML_Tensor/data/http/'
# FILES = glob.glob(HTML_DIR+'*/*.html')
#
# docs = []
#
# labels = []  # list of label ids
#
# cnt = len(FILES)
#
# for file in (FILES):
#
#     print(cnt,file)
#     cnt -= 1
#     path = file
#     label_name = file.split('/')[7]
#     if label_name =='ok':
#         label = 1
#     else:
#         label = 0
#     labels.append(label)
#     input = np.zeros(100)

    # print labels, 'labels'

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping



x_true = tf.placeholder(tf.float32, shape=[None, 250*300], name='x')

# The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x_true, [-1, 1, 250, 300])

# y = tf.placeholder(tf.float32, shape=[None,1], name='y')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
print (layer_conv1)




layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)



layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3)
print (num_features)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
# w1 = tf.Variable(tf.random_normal([300,1500],stddev=0.01))
# b1 = tf.Variable(tf.random_normal([1500]))
#
# w2 = tf.Variable(tf.random_normal([1500,300],stddev=0.01))
# b2 = tf.Variable(tf.random_normal([300]))
#
# w3 = tf.Variable(tf.random_normal([300,4],stddev=0.01))
# b3 = tf.Variable(tf.random_normal([4]))
#
# h1 = tf.add(tf.matmul(x_true,w1),b1)
# h1 = tf.nn.relu(h1)
#
# h2 = tf.add(tf.matmul(h1,w2),b2)
# h2 = tf.nn.relu(h2)
#
# output_layer = tf.add(tf.matmul(h2,w3),b3)
# output_layer = tf.nn.relu(output_layer)
#
# y_pred = tf.nn.softmax(output_layer)
#
# y_pred_cls = tf.argmax(y_pred, dimension=1)
#
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer,
#                                                         labels=y)
# cost = tf.reduce_mean(cross_entropy)
#
# optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#
# correct_prediction = tf.equal(y_pred_cls, y_true_cls)
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# session = tf.Session()
# session.run(tf.global_variables_initializer())
# session.run(tf.initialize_variables)
train_batch_size = batch_size

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):
        # print ('in optimize total iteration', i, total_iterations, str(int(data.train.num_examples / batch_size)))
        # Get a batch of training examples.
        # x_batch now holds a batch of docs and
        # y_true_batch are the true labels for those docs.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
       # print('len of x batch is',len(x_batch))
        #print cls_batch
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]
        x_batch = x_batch.reshape(train_batch_size, 250*300)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, 250*300)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x_true: x_batch,
                           y_true: y_true_batch}

        feed_dict_validate = {x_true: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        # print (data.train.num_examples, batch_size)
        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples / batch_size) == 0:
            # print("I AM IN IF", i, data.train.num_examples, batch_size, data.train.num_examples / batch_size)
    #      print i % int(data.train.num_examples / batch_size)

            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))

            acc = session.run(accuracy, feed_dict=feed_dict_train)
            val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
            train_loss = session.run(cost, feed_dict=feed_dict_train)
            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f} -- Acc Loss: {4:.3f}"
            print(msg.format(epoch + 1, acc, val_acc, val_loss, train_loss))


            # print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break
#        else:
 #           print("I AM IN ELSE", i , data.train.num_examples, batch_size, data.train.num_examples/batch_size)
  #          print i % int(data.train.num_examples / batch_size)


    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))




def print_test_accuracy(show_example_errors=False):
    # Number of docs in the test-set.
    num_test = len(data_test.test.docs)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        batch_size_test = 3
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size_test, num_test)

        # Get the docs from the test-set between index i and j.
        docs = data_test.test.docs[i:j, :].reshape(batch_size_test, 250*300)

        # Get the associated labels.
        labels = data_test.test.labels[i:j, :]

        # Create a feed-dict with these docs and labels.
        feed_dict = {x_true: docs, y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data_test.test.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified docs.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # docs divided by the total number of docs in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set Real: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    precision_recall_blockpage(y_pred=cls_pred, y_true=cls_true)
    print("_________________________________________________________")
    precision_recall_servererr(y_pred=cls_pred, y_true=cls_true)
    print("_________________________________________________________")
    precision_recall_connectionerr(y_pred=cls_pred, y_true=cls_true)
    print("_________________________________________________________")

def print_validation_accuracy(show_example_errors=False,
                              show_confusion_matrix=False):
    # Number of docs in the test-set.
    num_test = len(data.valid.docs)
    validation_batch_size = 16


    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + validation_batch_size, num_test)

        # Get the docs from the test-set between index i and j.
        docs = data.valid.docs[i:j, :].reshape(validation_batch_size, 250*300)


        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these docs and labels.
        feed_dict = {x_true: docs, y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified docs.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # docs divided by the total number of docs in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

optimize(num_iterations=1)
print_validation_accuracy()
print("Going for 9 more iterations")
optimize(num_iterations=9)  # We already performed 1 iteration above.
print_validation_accuracy(show_example_errors=True)
print(" 200 MOOOOOOORE ")
optimize(num_iterations=200)  # We already performed 1 iteration above.
print_validation_accuracy(show_example_errors=True)
print(" 200 MOOOOOOORE ")
optimize(num_iterations=4000)  # We already performed 1 iteration above.
print("Test Set")
print_test_accuracy(show_example_errors=True)
