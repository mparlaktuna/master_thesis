"""tests deeplda algorithm with tensorflow"""

import data_loader
import tensorflow as tf


def main():
    x = tf.placeholder(tf.float32, [None, 784], "x")
    model = create_network(x)



def create_network(x):
    weights = {
    "w1" : tf.Variable(tf.random_normal([3,3,1,64])),
    "w2" : tf.Variable(tf.random_normal([3,3,64,64])),
    "w3" : tf.Variable(tf.random_normal([3,3,64,96])),
    "w4" : tf.Variable(tf.random_normal([3,3,96,96])),
    "w5" : tf.Variable(tf.random_normal([3,3,96,256])),
    "w6" : tf.Variable(tf.random_normal([3,3,256,256])),
    "w7" : tf.Variable(tf.random_normal([1,1,256,10])),
    }

    #convolutional layers
    _x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d("conv1", _x, weights["w1"])
    norm1 = batch_norm(conv1)
    conv2 = conv2d("conv2", norm1, weights["w2"])
    norm2 = batch_norm(conv2)
    pool1 = max_pool("pool1", norm2, 2)
    drop1 = tf.nn.dropout(pool1, 0.25)

    conv3 = conv2d("conv3", drop1, weights["w3"])
    norm3 = batch_norm(conv3)
    conv4 = conv2d("conv4", norm3, weights["w4"])
    norm4 = batch_norm(conv4)
    pool2 = max_pool("pool2", norm4, 2)
    drop2 = tf.nn.dropout(pool2, 0.25)

    conv5 = conv2d("conv5", drop2, weights["w5"])
    norm5 = batch_norm(conv5)
    drop3 = tf.nn.dropout(norm5, 0.5)
    conv6 = conv2d("conv6", drop3, weights["w6"])
    norm6 = batch_norm(conv6)
    drop4 = tf.nn.dropout(norm6, 0.5)

    #classification layers
    conv7 = conv2d("conv7", drop4, weights["w7"])
    norm7 = batch_norm(conv7)
    out = tf.reshape(norm7, [-1, 7*7*10])

    return out

def cost_function(model):


def conv2d(name, l_input, w):
    return tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def batch_norm(input):
    return tf.contrib.layers.batch_norm(input)

if __name__ == '__main__':
    main()
