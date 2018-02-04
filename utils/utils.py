# -*- coding: utf-8 -*-
# @Time    : 2018/2/3 15:50
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : utils.py
# @Software: PyCharm

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.misc


mean_value = np.array([123.68, 116.779, 103.939])
def preprocess(image):
    return image - mean_value

def undo_preprocess(image):
    return image + mean_value

def load_image(image_path, size=None):
    image = PIL.Image.open(image_path)

    if size is not None:
        image = image.resize(size, PIL.Image.LANCZOS)

    return np.float32(image)


def generate_noise_image(content_image, noise_ratio=0.6):

    noise_image = np.random.uniform(-20, 20,(1, content_image.shape[1],content_image.shape[2], 3)).astype('float32')
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


def save_image(path, image):
    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)
    scipy.misc.imsave(path, image)




def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """

    m, n_H, n_W, n_C = tf.convert_to_tensor(a_G).get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, [-1, n_C])
    a_G_unrolled = tf.reshape(a_G, [-1, n_C])
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))) / (4 * n_H * n_W * n_C)
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(tf.transpose(A, [1, 0]), A)
    return GA

# GRADED FUNCTION: compute_layer_style_cost
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    ### START CODE HERE ###
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [-1, n_C])
    a_G = tf.reshape(a_G, [-1, n_C])
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / pow(2 * n_H * n_W * n_C, 2)

    return J_style_layer

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def compute_style_cost(out_S, out_G, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        J_style_layer = compute_layer_style_cost(out_S[layer_name], out_G[layer_name])

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

# GRADED FUNCTION: total_cost
def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + beta * J_style
    return J

