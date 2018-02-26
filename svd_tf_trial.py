import tensorflow as tf
import numpy as np
#from tensorflow.python import debug as tf_debug
import tools as tools

d = 784
x = tf.placeholder(tf.float32, [d, d], "x")
x2 = tf.placeholder(tf.float32, [d, d], "x2")
# Define loss and optimizer

s, u, v = tf.svd(x, full_matrices=True, compute_uv=True, name="svd")

W_conv1 = tools.weight_variable([5, 5, 1, 32], "w1")
b_conv1 = tools.bias_variable([32], "b1")
#
x_image = tf.reshape(u, [-1, 28, 28, 1])
#
h_conv1 = tf.nn.relu(tools.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = tools.max_pool_2x2(h_conv1)

W_conv2 = tools.weight_variable([5, 5, 32, 16], "w1")
b_conv2 = tools.bias_variable([16], "b1")
#
h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = tools.max_pool_2x2(h_conv2)

print("hconv1", h_pool2.shape)
h_pool2_flat = tf.reshape(h_pool2, [d, 7 * 7 * 16])

print("hflat1", h_pool2_flat.shape)
# h_pool1 = tools.max_pool_2x2(h_conv1)
#
# W_conv2 = tools.weight_variable([5, 5, 32, 64], "w2")
# b_conv2 = tools.bias_variable([64], "b2")
#
# h_conv2 = tf.nn.relu(tools.conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = tools.max_pool_2x2(h_conv2)
#
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# x_image_2 = tf.reshape(x2, [-1, 28, 28, 1])
#
# h_conv1_2 = tf.nn.relu(tools.conv2d(x_image_2, W_conv1) + b_conv1)
# h_pool1_2 = tools.max_pool_2x2(h_conv1_2)
#
# h_conv2_2 = tf.nn.relu(tools.conv2d(h_pool1_2, W_conv2) + b_conv2)
# h_pool2_2 = tools.max_pool_2x2(h_conv2_2)
#
# h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7 * 7 * 64])

#self.logger.info("First angle: {}".format(tools.calculateAngles(self.first_data, self.second_data, self.rank)))
# h_pool2 = tf.slice(h_pool2_flat, [0,0],[2400,2400])

#h_pool2 = tf.pad(h_pool2_flat, padding, "CONSTANT")

#b1 = tools.bias_variable((d,d))
#l1 = x + b1

P = tf.matmul(tf.transpose(h_pool2_flat, name="transpose1"), x2, name="matmul1")

p_diag = tf.diag_part(P, name="diagonal")
angles = tf.acos(p_diag, name="arccos")
sine = tf.sin(angles, name="sine")
total = d-tf.reduce_sum(tf.square(sine, name="square"), name="reduce_sum")
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(total)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

a = np.random.rand(d,d)
b = np.random.rand(d,d)
with tf.Session() as sess:

    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    #out = sess.run(h_pool2_flat, feed_dict={x: a, x2: b})
    #print(np.count_nonzero(out))

    #print(out.size)
    #print(out)
    #for i,second_data in self.clustered.items():
    for i in range(2000):

        sess.run(train_step,feed_dict={x: a, x2: b})
        #if i%10==0:
        out = sess.run(total, feed_dict={x: a, x2: b})
        print(out)
    #sess.run(train_step, feed_dict={x: self.first_data, x2: u2, keep_prob: 0.5})
    #out3 = sess.run(total, feed_dict={x: self.first_data, x2: u2, keep_prob: 1.0})
    #self.logger.info("for {} total {}".format(i, out3))

