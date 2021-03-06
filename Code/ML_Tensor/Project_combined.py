"""
Arian Niaki
Dec 2017, UMass Amherst
This is the main class for running the supervised learning algorithm for the Image and HTMLs combined
"""



import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datasets_combined
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

def precision_recall_ok(y_true, y_pred):
    num_correctly_detected_blockpages = 0
    detected_blockpage = 0
    true_blockpage = 0
    for i in range(len(y_true)):
        if(y_pred[i]=='ok' and y_true[i]=='ok'):
            num_correctly_detected_blockpages += 1

        if(y_pred[i]=='ok'):
            detected_blockpage += 1
        if(y_true[i]=='ok'):
            true_blockpage += 1
    print("num correctly detected ok ", num_correctly_detected_blockpages)
    print("detected ok ", detected_blockpage)
    print("true ok ", true_blockpage)
    try:
        precision = (num_correctly_detected_blockpages+0.0) / (detected_blockpage+0.0)
        recall = (num_correctly_detected_blockpages+0.0) / (true_blockpage+0.0)
        f1 = 2.0*precision*recall/(precision+recall+0.0)
        print("precision is : %s  recall is : %s , f1 score is for ok: %s" %(str(precision),str(recall),str(f1)))
    except Exception as exp:
        print("exception ",str(exp))


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
        f1 = 2.0*precision*recall/(precision+recall+0.0)
        print("precision is : %s  recall is : %s , f1 score is for Blockpage: %s" %(str(precision),str(recall),str(f1)))
    except Exception as exp:
        print("exception ",str(exp))
def precision_recall_servererr(y_true, y_pred):
    num_correctly_detected_servererrs = 0
    detected_servererr = 0
    true_servererr = 0
    for i in range(len(y_true)):
        if(y_pred[i]=='servererr' and y_true[i]=='servererr'):
            num_correctly_detected_servererrs += 1

        if(y_pred[i]=='servererr'):
            detected_servererr += 1
        if(y_true[i]=='servererr'):
            true_servererr += 1
    print("num correctly detected servererrs ", num_correctly_detected_servererrs)
    print("detected servererrs ", detected_servererr)
    print("true servererrs ", true_servererr)
    try:
        precision = (num_correctly_detected_servererrs+0.0) / (detected_servererr+0.0)
        recall = (num_correctly_detected_servererrs+0.0) / (true_servererr+0.0)

        f1 = 2.0*precision*recall/(precision+recall+0.0)
        print("precision is : %s  recall is : %s , f1 score is for Server: %s" %(str(precision),str(recall),str(f1)))
    except Exception as exp:
        print("exception ",str(exp))

def precision_recall_connectionerr(y_true, y_pred):
    num_correctly_detected_connectionerrs = 0
    detected_connectionerr = 0
    true_connectionerr = 0
    for i in range(len(y_true)):
        if(y_pred[i]=='connectionerr' and y_true[i]=='connectionerr'):
            num_correctly_detected_connectionerrs += 1

        if(y_pred[i]=='connectionerr'):
            detected_connectionerr += 1
        if(y_true[i]=='connectionerr'):
            true_connectionerr += 1
    print("num correctly detected connectionerrs ", num_correctly_detected_connectionerrs)
    print("detected connectionerrs ", detected_connectionerr)
    print("true connectionerrs ", true_connectionerr)
    try:
        precision = (num_correctly_detected_connectionerrs+0.0) / (detected_connectionerr+0.0)
        recall = (num_correctly_detected_connectionerrs+0.0) / (true_connectionerr+0.0)

        f1 = 2.0*precision*recall/(precision+recall+0.0)
        print("precision is : %s  recall is : %s , f1 score for Connection : %s" %(str(precision),str(recall),str(f1)))
    except Exception as exp:
        print("exception ",str(exp))




def plot_images(images, cls_true, cls_pred=None):
    if len(images) == 0:
        print("no images to show")
        return
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 4))

    images, cls_true = zip(*[(images[i], cls_true[i]) for i in random_indices])

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        if(i<len(images)):
            ax.imshow(images[i].reshape(img_size, img_size, num_channels))

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()





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

html_image_size = fc_size + fc_size

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3
num_channels_http = 300


# image dimensions (only squares for now)
img_size = 128

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['block', 'ok','servererr','connectionerr']
num_classes = len(classes)

# batch size
#batch_size = 539
batch_size = 21

# validation split
validation_size = .25

# how long to wait after validation loss stops improving before terminating training
early_stopping = None  # use None if you don't want to implement early stoping

train_path = 'data/train/'
test_path = 'data/test/'
# train_path = '/scratch/screenshots2/train/'
# test_path = '/scratch/screenshots2/test/'
checkpoint_dir = "models/"


data = datasets_combined.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
data_test = datasets_combined.read_test_set_labeled(test_path, img_size, classes)


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data_test.test.labels)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))


images, cls_true = data.train.images, data.train.cls

plot_images(images=images,cls_true=cls_true)

# HTML NET
x_true_html = tf.placeholder(tf.float32, shape=[None, 200*300], name='x')

# The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
x_image_html = tf.reshape(x_true_html, [-1, 1, 200, 300])



#
# x_true_html = tf.placeholder(tf.float32, shape=[None, 300], name='x_html')
# # y = tf.placeholder(tf.float32, shape=[None,1], name='y')
y_html = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true_html')


y_true_cls_html = tf.argmax(y_html, dimension=1)

dropout_param = tf.placeholder(tf.float32, name='dropout')



# END HTML NET
# First we define the placeholder variable for the input images. This allows us to change the images that are input to the TensorFlow graph. This is a so-called tensor, which just means that it is a multi-dimensional vector or matrix.
x_true = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
#

# The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
x_image = tf.reshape(x_true, [-1, img_size, img_size, num_channels])

# Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder variable x
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

# We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax
y_true_cls = tf.argmax(y_true, dimension=1)




layer_conv1_html, weights_conv1 = \
    new_conv_layer(input=x_image_html,
                   num_input_channels=num_channels_http,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
print (layer_conv1_html)

layer_conv2_html, weights_conv2 = \
    new_conv_layer(input=layer_conv1_html,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)



layer_conv3_html, weights_conv3 = \
    new_conv_layer(input=layer_conv2_html,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv3_html)
print (num_features)

layer_fc1_html = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)
layer_fc1_html = tf.nn.dropout(layer_fc1_html, keep_prob=dropout_param)

layer_fc2_html = new_fc_layer(input=layer_fc1_html,
                         num_inputs=fc_size,
                         num_outputs=fc_size,
                         use_relu=True)




# w1 = tf.Variable(tf.random_normal([300,1500],stddev=0.01))
# b1 = tf.Variable(tf.random_normal([1500]))
#
# w2 = tf.Variable(tf.random_normal([1500,300],stddev=0.01))
# b2 = tf.Variable(tf.random_normal([300]))

# w3 = tf.Variable(tf.random_normal([300,2],stddev=0.01))
# b3 = tf.Variable(tf.random_normal([2]))

# h1 = tf.add(tf.matmul(x_true_html,w1),b1)
# h1 = tf.nn.relu(h1)
#
# h2 = tf.add(tf.matmul(h1,w2),b2)
# html_output = tf.nn.relu(h2)
html_output = layer_fc2_html
# output_layer_html = tf.add(tf.matmul(h2,w3),b3)
# output_layer_html = tf.nn.relu(output_layer_html)

# y_pred_html = tf.nn.softmax(output_layer_html)

# y_pred_cls_html = tf.argmax(y_pred_html, dimension=1)

# cross_entropy_html = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer_html,
#                                                         labels=y_html)
# cost_html = tf.reduce_mean(cross_entropy_html)

# optimizer_html = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost_html)

# correct_prediction_html = tf.equal(y_pred_cls_html, y_true_cls_html)

# accuracy_html = tf.reduce_mean(tf.cast(correct_prediction_html, tf.float32))









layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)
print (layer_conv1)


layer_conv1 = tf.nn.dropout(layer_conv1, keep_prob=dropout_param)


layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)
layer_conv2 = tf.nn.dropout(layer_conv2, keep_prob=dropout_param)



layer_conv3, weights_conv3 = \
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

# layer_conv3 = tf.nn.dropout(layer_conv3, keep_prob=dropout_param)


layer_flat, num_features = flatten_layer(layer_conv3)
print (num_features)

image_output = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)



html_image = tf.concat([html_output, image_output], axis=1)



layer_fc_concat = new_fc_layer(input=html_image,
                         num_inputs=html_image_size,
                         num_outputs=html_image_size+100,
                         use_relu=False)


layer_fc2 = new_fc_layer(input=layer_fc_concat,
                         num_inputs=html_image_size+100,
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
# session.run(tf.initialize_variables)
train_batch_size = batch_size


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    global session
    # Calculate the accuracy on the training-set.
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    train_loss = session(cost, feed_dict=feed_dict_train)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f} -- Acc Loss: {4:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss, train_loss))


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
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, x_batch_html, y_true_batch, x_labels, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, x_valid_batch_html, y_valid_batch, x_valid_labels, valid_cls_batch = data.valid.next_batch(train_batch_size)
        # print('len of x batch is',len(x_batch))

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch_html = x_batch_html.reshape(train_batch_size, 200*300)
        x_valid_batch_html = x_valid_batch_html.reshape(train_batch_size, 200*300)



        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x_true: x_batch,
                           y_true: y_true_batch,
                           x_true_html: x_batch_html,
                           y_html: y_true_batch,
                           dropout_param: 0.5}

        feed_dict_validate = {x_true: x_valid_batch,
                              y_true: y_valid_batch,
                              x_true_html: x_valid_batch_html,
                              y_html: y_valid_batch,
                              dropout_param: 1.0
                              }

        # session.run(html_output, feed_dict=feed_dict_train_html)
        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples / batch_size) == 0:
      #      print("I AM IN IF", i , data.train.num_examples, batch_size, data.train.num_examples/batch_size)
            print i % int(data.train.num_examples / batch_size)

            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples / batch_size))

            acc = session.run(accuracy, feed_dict=feed_dict_train)
            val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
            train_loss = session.run(cost, feed_dict=feed_dict_train)
            msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f} -- Training Loss: {4:.3f}  "
            print(msg.format(epoch + 1, acc, val_acc, val_loss,train_loss))


            # print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)

            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


def plot_example_errors(cls_pred, correct):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.valid.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.valid.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:4],
                cls_true=cls_true[0:4],
                cls_pred=cls_pred[0:4])



def plot_example_errors_test(cls_pred, correct):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data_test.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data_test.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:4],
                cls_true=cls_true[0:4],
                cls_pred=cls_pred[0:4])




def plot_confusion_matrix(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.valid.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def print_test_accuracy(show_example_errors=False):
    # Number of images in the test-set.
    num_test = len(data_test.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        batch_size_test = 37
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size_test, num_test)


        # Get the images from the test-set between index i and j.
        images = data_test.test.images[i:j, :].reshape(batch_size_test, img_size_flat)
        docs = data_test.test.docs[i:j, :].reshape(batch_size_test, 200*300)
        # Get the associated labels.
        labels = data_test.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x_true: images, y_true: labels,
                     x_true_html: docs, y_html: labels, dropout_param: 1.0}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data_test.test.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
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
    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors_test(cls_pred=cls_pred, correct=correct)

def print_validation_accuracy(show_example_errors=False,
                              show_confusion_matrix=False):
    # Number of images in the test-set.
    validation_batch_size = 10
    num_test = len(data.valid.images)

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

        # Get the images from the test-set between index i and j.
        images = data.valid.images[i:j, :].reshape(validation_batch_size, img_size_flat)
        docs = data_test.test.docs[i:j, :].reshape(validation_batch_size, 200*300)

        # Get the associated labels.
        labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x_true: images, y_true: labels,
                     x_true_html: docs, y_html: labels,dropout_param:1.0}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)

optimize(num_iterations=1)
print_validation_accuracy(show_confusion_matrix=True)
optimize(num_iterations=199)  # We already performed 1 iteration above.
print_validation_accuracy(show_example_errors=True,show_confusion_matrix=True)
print("Test Set")
print_test_accuracy(show_example_errors=True)

# optimize(num_iterations=900)  # We performed 100 iterations above.
# print_validation_accuracy(show_example_errors=True)
# optimize(num_iterations=9000) # We performed 1000 iterations above.
# print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)
session.close()