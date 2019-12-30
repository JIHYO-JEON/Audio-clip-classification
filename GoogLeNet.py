import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from batch import minibatchTrain, minibatchTest
import matplotlib as plt
import numpy as np
import h5py




imageSize = 32
classNum = 10
trainNum = 50000
testNum = 10000


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


def inception_module(X, conv_1_size, conv_3_reduce_size, conv_3_size, conv_5_reduce_size, conv_5_size, pool_proj_size, name = 'inception'):
    with tf.variable_scope(name) as scope:
        conv_1 = conv_layer(X, filter_height=1, filter_width=1, num_filters=conv_1_size, name='{}_1x1'.format(name))
        conv_3_reduce = conv_layer(X, filter_height=1, filter_width=1, num_filters=conv_3_reduce_size, name='{}_3x3_reduce'.format(name))
        conv_3 = conv_layer(conv_3_reduce, filter_height=3, filter_width=3, num_filters=conv_3_size, name='{}_3x3'.format(name))
        conv_5_reduce = conv_layer(X, filter_height=1, filter_width=1, num_filters=conv_5_reduce_size, name='{}_5x5_reduce'.format(name))
        conv_5 = conv_layer(conv_5_reduce, filter_height=5, filter_width=5, num_filters=conv_5_size, name='{}_5x5'.format(name))
        pool = max_pool(X, stride=1, padding='SAME', name='{}_pool'.format(name))
        pool_proj = conv_layer(pool, filter_height=1, filter_width=1, num_filters=pool_proj_size, name='{}_pool_proj'.format(name))
        return tf.concat([conv_1,conv_3,conv_5,pool_proj], axis=3, name='{}_concat'.format(name))


tf.reset_default_graph()

# Create Placeholder
X = tf.placeholder(tf.float32, [None, imageSize, imageSize, 3])
Y = tf.placeholder(tf.float32, [None, classNum])
p = tf.placeholder(tf.float32)

# Model
conv1 = conv_layer(X, filter_height=7, filter_width=7, num_filters=64, stride=2, name='conv1')
pool1 = max_pool(conv1, name='pool1')
conv2 = conv_layer(pool1, filter_height=3, filter_width=3, num_filters=192, stride=1, name='conv2')
pool2 = max_pool(conv2, name='pool2')
inception3a = inception_module(pool2, 64, 96, 128, 16, 32, 32, name='inception3a')
inception3b = inception_module(inception3a, 128, 128, 192, 32, 96, 64, name='inception3b')
pool3 = max_pool(inception3a, name='pool3')
inception4a = inception_module(pool3, 192, 96, 208, 16, 48, 64, name='inception4a')
inception4b = inception_module(inception4a, 160, 112, 224, 24, 64, 64, name='inception4b')
inception4c = inception_module(inception4b, 128, 128, 256, 24, 64, 64, name='inception4c')
inception4d = inception_module(inception4c, 112, 144, 288, 32, 64, 64, name='inception4d')
inception4e = inception_module(inception4d, 256, 160, 320, 32, 128, 128, name='inception4e')
pool4 = max_pool(inception4e, name='pool4')
inception5a = inception_module(pool4, 256, 160, 320, 32, 128, 128, name='inception5a')
inception5b = inception_module(inception5a, 384, 192, 384, 48, 128, 128, name='inception5b')
avg_pool = tf.nn.avg_pool(inception5b, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME', name='avg_pool')
flatten = tf.contrib.layers.flatten(avg_pool)
fcl1 = fcl(flatten, 1024, p)
fcl2 = fcl(fcl1, 1000, 1.0)
fcl3 = fcl(fcl2, classNum, 1.0, activation_fn=tf.nn.softmax)
GoogLeNet = fcl3

# loss & train
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = GoogLeNet, labels = Y))
train = tf.train.AdamOptimizer(0.0001).minimize(loss)

# accuracy
correct_prediction = tf.equal(tf.argmax(GoogLeNet,1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# run
'''
hdf5Path ='dogs.hdf5'
dataset = h5py.File(hdf5Path, 'r')
totalImages = dataset['trainImages']
totallabels = np.array(dataset['trainLabels']).reshape(trainNum, -1)
totallabels = np.eye(classNum)[totallabels.reshape(-1)]
'''

(x_train, y_train), (x_test, y_test) = load_data()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test,10), axis=1)
costs = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Restore the network
    if tf.train.get_checkpoint_state(recall_path):  # backup 폴더 만들기/ 이전까지 만든 checkpoint가 있는가?
        saver = tf.train.Saver()  # 저장가능한 객체를 만들기
        saver.restore(sess, recall_path)  # backup폴더는 코드파일 위치에
        print("good!")

    # training
    for epoch in range(10000):# 트레이닝을 얼마나 시킬지
        total_cost = 0

        for i in range(total_batch):
            batch = next_batch(train_batch_size, x_train, y_train_one_hot.eval())
            _, temp_cost = sess.run((train, loss), feed_dict={X: batch[0], Y: batch[1], p: 0.4})
            total_cost = total_cost + temp_cost
        print('Epoch:', '%04d' % (epoch+1), 'Avg. cost =', '{:.6f}'.format(total_cost / total_batch))

        if epoch % 5 == 0:
            train_acc = accuracy.eval({X: batch[0], Y: batch[1], p: 1.0})
            print("Train Accuracy:", train_acc)
            saver = tf.train.Saver()
            saver.save(sess, recall_path, write_meta_graph=False)

        if epoch % 1 == 0:
            # collect temp_cost for ploting the cost convergence figure
            costs.append(total_cost)

    #test after training
    test_acc = 0.0
    test_epoch = 10
    for i in range(total_batch_test):
        test_batch = next_batch(test_batch_size, x_test, y_test_one_hot.eval())
        test_acc = test_acc + accuracy.eval(feed_dict={X: test_batch[0], Y: test_batch[1], p: 1.0})
    test_acc = test_acc/test_epoch
    print('Test Accuracy: %f' %test_acc)