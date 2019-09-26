import tensorflow as tf


def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def create_norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def mnist_base(x, y, pro):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    features = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    fcc_weights = {
        'W_fc1': weight_variable([7 * 7 * 64, 256]),
        'b_fc1': bias_variable([256]),
    }
    norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])
    #norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']))
    norm_weighted_mean = tf.reduce_sum(norm_weighted, axis=1)
    wrong_norm = 1 / norm_weighted_mean
    #
    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted


def mnist_base_layer_norm(x, y, pro):
    t = 64
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([3, 3, 1, 16])
    b_conv1 = bias_variable([16])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)
    l_norm1 = tf.contrib.layers.layer_norm(h_pool1)

    W_conv2 = weight_variable([3, 3, 16, 32])
    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(l_norm1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)
    l_norm2 = tf.contrib.layers.layer_norm(h_pool2)

    W_conv3 = weight_variable([3, 3, 32, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(l_norm2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3)
    l_norm3 = tf.contrib.layers.layer_norm(h_pool3)

    W_conv4 = weight_variable([3, 3, 64, 128])
    b_conv4 = bias_variable([128])

    h_conv4 = tf.nn.relu(conv2d(l_norm3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4)
    l_norm4 = tf.contrib.layers.layer_norm(h_pool4)

    W_conv5 = weight_variable([2, 2, 128, 256])
    b_conv5 = bias_variable([256])

    h_conv5 = tf.nn.relu(conv2d(l_norm4, W_conv5) + b_conv5)
    h_pool5 = max_pool(h_conv5)
    #l_norm5 = tf.contrib.layers.layer_norm(h_pool5)

    features = tf.reshape(h_pool5, [-1, 1 * 1 * 256])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

    rejections = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    fcc_weights = {
        'W_fc1': weight_variable([256, 256]),
        'b_fc1': bias_variable([256]),
    }

    rejection_weighted = tf.nn.relu(tf.matmul(rejections, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])

    rejection = tf.reduce_sum(tf.square(rejection_weighted), axis=1)
    wrong_rejection = 1 / rejection

    norm_minimize = tf.where(tf.greater(y, 0), rejection, wrong_rejection)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, rejection, tf_training_model, rejections


def mnist_conv1(x, y, pro):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3)

    features = tf.reshape(h_pool3, [-1, 4 * 4 * 64])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    fcc_weights = {
        'W_fc1': weight_variable([4 * 4 * 64, 1]),
        'b_fc1': bias_variable([128]),
        # 'W_fc2': weight_variable([512, 1]),
        # 'b_fc2': bias_variable([1]),

    }
    #norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])
    #norm_weighted = tf.nn.relu(tf.matmul(norm_weighted, fcc_weights['W_fc2']) + fcc_weights['b_fc2'])
    norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']))
    norm_weighted_mean = tf.reduce_sum(norm_weighted, axis=1)
    wrong_norm = 1 / norm_weighted_mean
    #
    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted


def mnist_conv2(x, y, pro):
    t = 32
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([3, 3, 1, t])
    b_conv1 = bias_variable([t])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([3, 3, t, t])
    b_conv2 = bias_variable([t])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    W_conv3 = weight_variable([3, 3, t, t])
    b_conv3 = bias_variable([t])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3)

    W_conv4 = weight_variable([3, 3, t, t])
    b_conv4 = bias_variable([t])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4)

    W_conv5 = weight_variable([3, 3, t, t])
    b_conv5 = bias_variable([t])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool(h_conv5)

    features = tf.reshape(h_pool5, [-1, 1 * 1 * t])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    fcc_weights = {
        'W_fc1': weight_variable([t, 32]),
        'b_fc1': bias_variable([32]),
    }
    norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])
    #norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']))
    norm_weighted_mean = tf.reduce_sum(norm_weighted, axis=1)
    wrong_norm = 1 / norm_weighted_mean
    #
    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted


def mnist_only_conv(x, y, pro):
    t = 128
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = weight_variable([3, 3, 1, t])
    b_conv1 = bias_variable([t])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([3, 3, t, t])
    b_conv2 = bias_variable([t])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    W_conv3 = weight_variable([3, 3, t, t])
    b_conv3 = bias_variable([t])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3)

    W_conv4 = weight_variable([3, 3, t, t])
    b_conv4 = bias_variable([t])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4)

    W_conv5 = weight_variable([3, 3, t, t])
    b_conv5 = bias_variable([t])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool(h_conv5)

    features = tf.reshape(h_pool5, [-1, 1 * 1 * t])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    norm_weighted_mean = tf.reduce_sum(norm, axis=1)
    wrong_norm = 1 / norm_weighted_mean

    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted

def cifar10_base(x,y,pro):
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    features = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    #features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))
    features_norm_tra = tf.transpose(features)

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    fcc_weights = {
        'W_fc1': weight_variable([8*8*64, 256]),
        'b_fc1': bias_variable([256]),
    }

    norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])

    norm_weighted_mean = tf.reshape(tf.reduce_sum(norm_weighted, axis=1), [-1,1])
    wrong_norm = 1 / norm_weighted_mean

    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted


def cifar10_conv(x,y,pro, keep_prob):
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
    h_pool1 = max_pool(h_conv1)
    #l_norm1 = tf.contrib.layers.layer_norm(h_pool1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob)
    h_pool2 = max_pool(h_conv2)
    #l_norm2 = tf.contrib.layers.layer_norm(h_pool2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_conv3 = tf.nn.dropout(h_conv3, keep_prob)
    h_pool3 = max_pool(h_conv3)
    #l_norm3 = tf.contrib.layers.layer_norm(h_pool3)

    W_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_conv4 = tf.nn.dropout(h_conv4, keep_prob)

    h_pool4 = max_pool(h_conv4)
    #l_norm4 = tf.contrib.layers.layer_norm(h_pool4)

    W_conv5 = weight_variable([3, 3, 64, 64])
    b_conv5 = bias_variable([64])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_conv5 = tf.nn.dropout(h_conv5, keep_prob)
    h_pool5 = max_pool(h_conv5)

    features = tf.reshape(h_pool5, [-1, 1 * 1 * 64])

    #features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))
    features_norm_tra = tf.transpose(features)

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))
    dropout = tf.layers.dropout(
        inputs=norm, rate=0.4)

    fcc_weights = {
        'W_fc1': weight_variable([64, 10]),
        'b_fc1': bias_variable([10]),
    }

    norm_weighted = tf.nn.relu(tf.matmul(dropout, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])

    norm_weighted_mean = tf.reshape(tf.reduce_sum(norm_weighted, axis=1), [-1,1])
    wrong_norm = 1 / norm_weighted_mean

    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted

def cifar10_conv2(x,y,pro, keep_prob):
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    W_conv1 = weight_variable([3, 3, 3, 128])
    b_conv1 = bias_variable([128])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_conv1 = tf.nn.dropout(h_conv1, keep_prob)
    h_pool1 = max_pool(h_conv1)
    #h_pool1 = tf.contrib.layers.layer_norm(h_pool1)


    W_conv2 = weight_variable([3, 3, 128,128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_conv2 = tf.nn.dropout(h_conv2, keep_prob)
    h_pool2 = max_pool(h_conv2)
    #h_pool2 = tf.contrib.layers.layer_norm(h_pool2)

    W_conv3 = weight_variable([3, 3, 128, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_conv3 = tf.nn.dropout(h_conv3, keep_prob)
    h_pool3 = max_pool(h_conv3)
    #h_pool3 = tf.contrib.layers.layer_norm(h_pool3)

    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4)
    #h_pool4 = tf.contrib.layers.layer_norm(h_pool4)

    W_conv5 = weight_variable([3, 3, 256, 512])
    b_conv5 = bias_variable([512])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool(h_conv5)

    features = tf.reshape(h_pool5, [-1, 1 * 1 * 512])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))
    #features_norm_tra = tf.transpose(features)

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))
    dropout = tf.layers.dropout(
        inputs=norm, rate=0.4)

    fcc_weights = {
        'W_fc1': weight_variable([512, 128]),
        'b_fc1': bias_variable([128]),
    }

    norm_weighted = tf.nn.relu(tf.matmul(dropout, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])

    norm_weighted_mean = tf.reshape(tf.reduce_sum(norm_weighted, axis=1), [-1,1])
    wrong_norm = 1 / norm_weighted_mean

    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted




def create_norm_vgg19(x, y, pro):
    x_image = tf.reshape(x, [-1, 32, 32, 3])

    cnn_weights = {
        'w1': weight_variable([3, 3, 3, 64]),
        'b1': bias_variable([64]),
        'w2': weight_variable([3, 3, 64, 64]),
        'b2': bias_variable([64]),
        'w3': weight_variable([3, 3, 64, 128]),
        'b3': bias_variable([128]),
        'w4': weight_variable([3, 3, 128, 128]),
        'b4': bias_variable([128]),
        'w5': weight_variable([3, 3, 128, 256]),
        'b5': bias_variable([256]),
        'w6': weight_variable([3, 3, 256, 256]),
        'b6': bias_variable([256]),
        'w7': weight_variable([3, 3, 256, 256]),
        'b7': bias_variable([256]),
        'w8': weight_variable([3, 3, 256, 512]),
        'b8': bias_variable([512]),
        'w9': weight_variable([3, 3, 512, 512]),
        'b9': bias_variable([512]),
        'w10': weight_variable([3, 3, 512, 512]),
        'b10': bias_variable([512]),
        'w11': weight_variable([3, 3, 512, 512]),
        'b11': bias_variable([512]),
        'w12': weight_variable([3, 3, 512, 512]),
        'b12': bias_variable([512]),
        'w13': weight_variable([3, 3, 512, 512]),
        'b13': bias_variable([512]),
    }

    # h_conv1 = tf.nn.relu(conv2d(x_image, cnn_weights["w1"]) + cnn_weights["b1"])
    # h_conv2 = tf.nn.relu(conv2d(h_conv1, cnn_weights["w2"]) + cnn_weights["b2"])
    # h_pool1 = max_pool(h_conv2)
    #
    # h_conv3 = tf.nn.relu(conv2d(h_pool1, cnn_weights["w3"]) + cnn_weights["b3"])
    # h_conv4 = tf.nn.relu(conv2d(h_conv3, cnn_weights["w4"]) + cnn_weights["b4"])
    # h_pool2 = max_pool(h_conv4)
    #
    # h_conv5 = tf.nn.relu(conv2d(h_pool2, cnn_weights["w5"]) + cnn_weights["b5"])
    # h_conv6 = tf.nn.relu(conv2d(h_conv5, cnn_weights["w6"]) + cnn_weights["b6"])
    # h_conv7 = tf.nn.relu(conv2d(h_conv6, cnn_weights["w7"]) + cnn_weights["b7"])
    # h_pool3 = max_pool(h_conv7)
    #
    # h_conv8 = tf.nn.relu(conv2d(h_pool3, cnn_weights["w8"]) + cnn_weights["b8"])
    # h_conv9 = tf.nn.relu(conv2d(h_conv8, cnn_weights["w9"]) + cnn_weights["b9"])
    # h_conv10 = tf.nn.relu(conv2d(h_conv9, cnn_weights["w10"]) + cnn_weights["b10"])
    # h_pool4 = max_pool(h_conv10)
    #
    # h_conv11 = tf.nn.relu(conv2d(h_pool4, cnn_weights["w11"]) + cnn_weights["b11"])
    # h_conv12 = tf.nn.relu(conv2d(h_conv11, cnn_weights["w12"]) + cnn_weights["b12"])
    # h_conv13 = tf.nn.relu(conv2d(h_conv12, cnn_weights["w13"]) + cnn_weights["b13"])
    # h_pool5 = max_pool(h_conv13)

    #features = tf.reshape(h_pool5, [-1, 512])

    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    features = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

    features_norm_tra = tf.transpose(tf.nn.l2_normalize(features, axis=1))

    norm = tf.transpose(features_norm_tra - tf.matmul(pro, features_norm_tra))

    fcc_weights = {
        'W_fc1': weight_variable([8*8*64, 256]),
        'b_fc1': bias_variable([256]),
        'W_fc2': weight_variable([4096, 256]),
        'b_fc2': bias_variable([256]),
    }

    norm_weighted = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])

    # norm_weighted1 = tf.nn.relu(tf.matmul(norm, fcc_weights['W_fc1']) + fcc_weights['b_fc1'])
    # norm_weighted = tf.nn.relu(tf.matmul(norm_weighted1, fcc_weights['W_fc2']) + fcc_weights['b_fc2'])

    norm_weighted_mean = tf.reshape(tf.reduce_sum(norm_weighted, axis=1), [-1,1])
    wrong_norm = 1 / norm_weighted_mean
    #
    norm_minimize = tf.where(tf.greater(y, 0), norm_weighted_mean, wrong_norm)

    tf_training_model = tf.train.AdamOptimizer(1e-4).minimize(norm_minimize)

    return features, norm_weighted_mean, tf_training_model, norm_weighted


model_details = {"mnist": {"base": {"feature_size" : 3136, "model": mnist_base},
                           "conv1": {"feature_size" : 1024, "model": mnist_conv1},
                           "conv2": {"feature_size" : 32, "model": mnist_conv2},
                           "only_conv": {"feature_size" : 128, "model": mnist_only_conv},
                           "layer_norm": {"feature_size" : 256, "model": mnist_base_layer_norm}
                           },
          "cifar10": {"base": {"feature_size":4096, "model": cifar10_base},
                      "conv": {"feature_size": 64, "model": cifar10_conv},
                      "conv2": {"feature_size": 512, "model": cifar10_conv2}}
          }

