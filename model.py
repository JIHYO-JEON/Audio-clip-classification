import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import h5py

def bringdata(filename):
    hdf5Path = filename
    dataset = h5py.File(hdf5Path, 'r')
    data = dataset['audio']
    labels = dataset['labels']
    print 'data brought', filename
    print data.shape, labels.shape
    return data, labels

def conv_layer(X, filter_height, filter_width, num_filters, name, stride = 1, padding = 'SAME'):
    input_channels = int(X.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
                                  dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(seed=0))
        biases = tf.get_variable('biases', shape=[num_filters], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.relu(tf.nn.conv2d(X, weights, strides=[1, stride, stride, 1], padding= padding) + biases)
        return conv

def max_pool(X, name, filter_height=3, filter_width=3, stride = 2, padding = 'SAME'):
    with tf.variable_scope(name) as scope:
        pool = tf.nn.max_pool(X, ksize=[1, filter_height, filter_width, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
        return pool

def fcl(X, output_size, dropout_ratio, activation_fn= tf.nn.relu):
    fc1 = tf.contrib.layers.fully_connected(X, output_size, activation_fn = activation_fn)
    fcl_drop = tf.nn.dropout(fc1, dropout_ratio)
    return fcl_drop


frames = 41
bands = 60
num_channels = 2

#feature_size = 2460 #60x41
num_labels = 10

X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

# Model
conv1 = conv_layer(X, filter_height=3, filter_width=3, num_filters=64, stride=1, name='conv1')
conv2 = conv_layer(conv1, filter_height=3, filter_width=3, num_filters=64, stride=1, name='conv2')
pool1 = max_pool(conv2, name='pool1')
conv3 = conv_layer(pool1, filter_height=3, filter_width=3, num_filters=128, stride=1, name='conv3')
conv4 = conv_layer(conv3, filter_height=3, filter_width=3, num_filters=128, stride=1, name='conv4')
pool2 = max_pool(conv4, name='pool2')
conv5 = conv_layer(pool2, filter_height=3, filter_width=3, num_filters=256, stride=1, name='conv5')
conv6 = conv_layer(conv5, filter_height=3, filter_width=3, num_filters=256, stride=1, name='conv6')
conv7 = conv_layer(conv6, filter_height=3, filter_width=3, num_filters=256, stride=1, name='conv7')
pool3 = max_pool(conv7, name='pool3')
conv8 = conv_layer(pool3, filter_height=3, filter_width=3, num_filters=512, stride=1, name='conv8')
conv9 = conv_layer(conv8, filter_height=3, filter_width=3, num_filters=512, stride=1, name='conv9')
conv10 = conv_layer(conv9, filter_height=3, filter_width=3, num_filters=512, stride=1, name='conv10')
pool4 = max_pool(conv10, name='pool4')
conv11 = conv_layer(pool4, filter_height=3, filter_width=3, num_filters=512, stride=1, name='conv11')
conv12 = conv_layer(conv11, filter_height=3, filter_width=3, num_filters=512, stride=1, name='conv12')
conv13 = conv_layer(conv12, filter_height=3, filter_width=3, num_filters=512, stride=1, name='conv13')
pool5 = max_pool(conv13, name='pool5')

flatten = tf.contrib.layers.flatten(pool5)
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
fcl1 = fcl(flatten, 1024, keep_prob1)
fcl2 = fcl(fcl1, 1000, keep_prob2)
fcl3 = fcl(fcl2, num_labels, 1.0, activation_fn=tf.nn.softmax)

# matplotlib inline
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 13


# loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = fcl3, labels = Y))
train_optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
cost_history = np.empty(shape=[1], dtype=float)

# accuracy
correct_prediction = tf.equal(tf.argmax(fcl3,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

'''
cost = -tf.reduce_sum(Y * tf.log(fcl3))
train_optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
correct_prediction = tf.equal(tf.argmax(fcl3, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cost_history = np.empty(shape=[1], dtype=float)
'''

batch_size = 10
iterations = 3000

traindata, trainlabel = bringdata('train')
testdata, testlabel = bringdata('test')

totalacc = 0

with tf.Session() as session:
    tf.initialize_all_variables().run()

    # Training
    for itr in range(iterations):

        c_sum = 0

        for i in range(48840/batch_size):
            offset = (i * batch_size) % (trainlabel.shape[0] - batch_size)
            batch_x = traindata[offset:(offset + batch_size), :, :, :]
            batch_y = trainlabel[offset:(offset + batch_size), :]

            _, c = session.run([train_optimizer, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob1: 1.0, keep_prob2: 1.0})
            c_sum += c / batch_size

        cost_history = np.append(cost_history, c_sum)

        print itr, '/', iterations, 'done', 'cost:', c

    fig = plt.figure(figsize=(15, 10))
    plt.plot(cost_history=True)
    plt.axis([0, iterations, 0, np.max(cost_history)])
    plt.show()

    # Test
    for itr in range(9):
        offset = (itr * batch_size) % (testlabel.shape[0] - batch_size)
        batch_x = testdata[offset:(offset + batch_size), :, :, :]
        batch_y = testlabel[offset:(offset + batch_size), :]

        acc = session.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob1: 1.0, keep_prob2: 1.0})
        totalacc += acc/9

    print 'Test accuracy:', totalacc