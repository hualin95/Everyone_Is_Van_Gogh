# -*- coding: utf-8 -*-
# @Time    : 2018/2/3 22:50
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : vgg19_tf.py
# @Software: PyCharm

'''
This is a implemention of VGG19 with weights initialized.
'''

import tensorflow as tf
import numpy as np

class VGG19:
    def __init__(self,VGG19_Model_Path = None):
        self.wDict = np.load(VGG19_Model_Path, encoding="bytes").item()



    def build(self,picture):
        self.conv1_1 = tf.nn.conv2d(
                        input=picture,
                        filter=self.wDict['conv1_1'][0],
                        strides=[1,1,1,1],
                        padding='SAME',
                        name='conv1_1'
                    )
        self.relu1_1 = tf.nn.relu(tf.nn.bias_add(self.conv1_1,self.wDict['conv1_1'][1]))

        self.conv1_2 = tf.nn.conv2d(
                        input=self.relu1_1,
                        filter=self.wDict['conv1_2'][0],
                        strides=[1,1,1,1],
                        padding='SAME',
                        name='conv1_1'
                    )
        self.relu1_2 = tf.nn.relu(tf.nn.bias_add(self.conv1_2, self.wDict['conv1_2'][1]))

        self.pool1 = tf.layers.max_pooling2d(
                     inputs=self.relu1_2,
                     pool_size=2,
                     strides=2,
                     name='pool1'
                     )


        # block 2
        self.conv2_1 = tf.nn.conv2d(
            input=self.pool1,
            filter=self.wDict['conv2_1'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv2_1'
        )
        self.relu2_1 = tf.nn.relu(tf.nn.bias_add(self.conv2_1, self.wDict['conv2_1'][1]))

        self.conv2_2 = tf.nn.conv2d(
            input=self.relu2_1,
            filter=self.wDict['conv2_2'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv2_2'
        )
        self.relu2_2 = tf.nn.relu(tf.nn.bias_add(self.conv2_2, self.wDict['conv2_2'][1]))
        self.pool2 = tf.layers.max_pooling2d(
            inputs=self.relu2_2,
            pool_size=2,
            strides=2,
            name='pool2'
        )

        # block 3
        self.conv3_1 = tf.nn.conv2d(
            input=self.pool2,
            filter=self.wDict['conv3_1'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_1'
        )
        self.relu3_1 = tf.nn.relu(tf.nn.bias_add(self.conv3_1, self.wDict['conv3_1'][1]))

        self.conv3_2 = tf.nn.conv2d(
            input=self.relu3_1,
            filter=self.wDict['conv3_2'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_2'
        )
        self.relu3_2 = tf.nn.relu(tf.nn.bias_add(self.conv3_2, self.wDict['conv3_2'][1]))
        self.conv3_3 = tf.nn.conv2d(
            input=self.relu3_2,
            filter=self.wDict['conv3_3'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_3'
        )
        self.relu3_3 = tf.nn.relu(tf.nn.bias_add(self.conv3_3, self.wDict['conv3_3'][1]))

        self.conv3_4 = tf.nn.conv2d(
            input=self.relu3_3,
            filter=self.wDict['conv3_4'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv3_4'
        )
        self.relu3_4 = tf.nn.relu(tf.nn.bias_add(self.conv3_4, self.wDict['conv3_4'][1]))
        self.pool3 = tf.layers.max_pooling2d(
            inputs=self.relu3_4,
            pool_size=2,
            strides=2,
            name='pool3'
        )

        # block 4
        self.conv4_1 = tf.nn.conv2d(
            input=self.pool3,
            filter=self.wDict['conv4_1'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_1'
        )
        self.relu4_1 = tf.nn.relu(tf.nn.bias_add(self.conv4_1, self.wDict['conv4_1'][1]))

        self.conv4_2 = tf.nn.conv2d(
            input=self.relu4_1,
            filter=self.wDict['conv4_2'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_2'
        )
        self.relu4_2 = tf.nn.relu(tf.nn.bias_add(self.conv4_2, self.wDict['conv4_2'][1]))
        self.conv4_3 = tf.nn.conv2d(
            input=self.relu4_2,
            filter=self.wDict['conv4_3'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_3'
        )
        self.relu4_3 = tf.nn.relu(tf.nn.bias_add(self.conv4_3, self.wDict['conv4_3'][1]))

        self.conv4_4 = tf.nn.conv2d(
            input=self.relu4_3,
            filter=self.wDict['conv4_4'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv4_4'
        )
        self.relu4_4 = tf.nn.relu(tf.nn.bias_add(self.conv4_4, self.wDict['conv4_4'][1]))
        self.pool4 = tf.layers.max_pooling2d(
            inputs=self.relu4_4,
            pool_size=2,
            strides=2,
            name='pool4'
        )


        # block 5
        self.conv5_1 = tf.nn.conv2d(
            input=self.pool4,
            filter=self.wDict['conv5_1'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_1'
        )
        self.relu5_1 = tf.nn.relu(tf.nn.bias_add(self.conv5_1, self.wDict['conv5_1'][1]))

        self.conv5_2 = tf.nn.conv2d(
            input=self.relu5_1,
            filter=self.wDict['conv5_2'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_2'
        )
        self.relu5_2 = tf.nn.relu(tf.nn.bias_add(self.conv5_2, self.wDict['conv5_2'][1]))
        self.conv5_3 = tf.nn.conv2d(
            input=self.relu5_2,
            filter=self.wDict['conv5_3'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_3'
        )
        self.relu5_3 = tf.nn.relu(tf.nn.bias_add(self.conv5_3, self.wDict['conv5_3'][1]))

        self.conv5_4 = tf.nn.conv2d(
            input=self.relu5_3,
            filter=self.wDict['conv5_4'][0],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv5_4'
        )
        self.relu5_4 = tf.nn.relu(tf.nn.bias_add(self.conv5_4, self.wDict['conv5_4'][1]))
        self.pool5 = tf.layers.max_pooling2d(
            inputs=self.relu5_4,
            pool_size=2,
            strides=2,
            name='pool5'
        )
        self.fc_in = tf.layers.flatten(self.pool5)
        self.fc6 = tf.layers.dense(
            inputs=self.fc_in,
            units=4096,
            activation=tf.nn.relu,
            name='fc6'
        )
        self.dropout1 = tf.layers.dropout(self.fc6,rate=0.5)
        self.fc7 = tf.layers.dense(
            inputs=self.dropout1,
            units=4096,
            activation=tf.nn.relu,
            name='fc7'
        )
        self.dropout2 = tf.layers.dropout(self.fc7, rate=0.5)
        self.fc8 = tf.layers.dense(
            inputs=self.dropout2,
            units=1000,
            activation=tf.nn.relu,
            name='fc8'
        )




