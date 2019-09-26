import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from settings import *
import os
from scipy.sparse.linalg import svds
from scipy import linalg
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import matplotlib


def createLoggers(file_name):
    """create logger with date and time
    also log what is run with this run
    """
    logger = logging.getLogger('logger_master')
    logger.setLevel(logging.DEBUG)

    if is_logging:
        fh = logging.FileHandler(log_location+file_name, 'w')
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)


def calculateSubspaceAngles(M1,M2):
    m1_orth = linalg.orth(M1)
    m2_orth = linalg.orth(M2)
    U, s, V = linalg.svd(np.matmul(np.transpose(m1_orth), m2_orth))
    costheta = np.flipud(s)
    theta = np.arccos(np.minimum(1, costheta))
    return theta


def plotSvd(M):
    """plots the u of svd(M)"""
    U, s, Vh = linalg.svd(M)
    diag = [U[x,x] for x in range(len(U))]
    diag_s = sorted(diag, reverse=True)
    plt.figure()
    plt.plot(range(len(s)), s)
    plt.show()


def calculateProjectionMatrix(M, rank=None):
    """return the projection matrix for matrix M"""
    if rank is None:
        U, s, Vh = linalg.svd(np.transpose(M))
    else:
        U, s, Vh = svds(np.transpose(M), rank)
    p = np.matmul(U, np.transpose(U))
    return p


def calculateNorm(v, P):
    """returns the norm of the vector v using the projection matrix P"""
    error = v - np.matmul(P, v)
    return np.sqrt(np.sum(np.square(error)))


def calculateAngles(M1, M2, rank=None):
    if rank:
        m1_u, s, v = svds(np.transpose(M1.astype(float)), k=rank)
        m2_u, s, v = svds(np.transpose(M2.astype(float)), k=rank)
    else:
        m1_u, s, v = linalg.svd(M1)
        m2_u, s, v = linalg.svd(M2)
    P = np.matmul(np.transpose(m1_u), m2_u)
    p_diag = np.diag(P)
    angles = np.arccos(p_diag)
    total = np.sum(np.square(np.sin(angles)))
    return total

def calculateIndividualAngles(M1, M2, rank=None):
    if rank:
        m1_u, s, v = svds(np.transpose(M1.astype(float)), k=rank)
        m2_u, s, v = svds(np.transpose(M2.astype(float)), k=rank)
    else:
        m1_u, s, v = linalg.svd(M1)
        m2_u, s, v = linalg.svd(M2)
    P = np.matmul(np.transpose(m1_u), m2_u)
    p_diag = np.diag(P)
    angles = np.sin(np.arccos(p_diag))
    return angles

def calculateAngleBetweenSvds(M1, rank=10):
    if rank:
        m1_u, s, v = svds(np.transpose(M1.astype(float)), k=rank)
        B = np.matmul(m1_u,np.diag(s))
        A = np.matmul(B,v)
        m1_u, s, v = svds(A.astype(float), k=rank)
        m2_u, s, v = svds(np.transpose(M1.astype(float)), k=rank)

    P = np.matmul(np.transpose(m1_u), m2_u)
    s = linalg.svd(P)
    angles = np.sin(np.arccos(s))
    return angles

def calculateAnglesAfterSvd(m1_u, m2_u):
    P = np.matmul(np.transpose(m1_u), m2_u)
    p_diag = np.diag(P)
    angles = np.arccos(p_diag)
    #total = np.sum(np.square(np.sin(angles)))
    return angles

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def convert_rgb_to_hsv_onlyh(images, sine=False):
    temp = []
    for j in images:
        if sine:
            temp.append(np.sin(np.deg2rad(matplotlib.colors.rgb_to_hsv(j)[:, :, 0:1])))
        else:
            temp.append(matplotlib.colors.rgb_to_hsv(j)[:,:,0:1])
    return np.array(temp)

def create_new_h_pool2_flat(x,w1,w2,b1,b2):
    #
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    #
    h_conv1 = tf.nn.relu(conv2d(x_image, w1) + b1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)

    h_pool2 = max_pool_2x2(h_conv2)
    return tf.reshape(h_pool2, [-1, 7 * 7 * 64])
