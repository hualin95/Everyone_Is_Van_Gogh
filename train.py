# -*- coding: utf-8 -*-
# @Time    : 2018/2/3 22:28
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train.py
# @Software: PyCharm


from utils.utils import *
from models.vgg19_tf import VGG19
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--VGG19_Model_Path',type=str,default='weight/vgg19.npy',help='The model_path of VGG19 where you store.')
parser.add_argument('--content_image',type=str,default='images/content/ucas.jpg',help='The path of content_image where you store.')
parser.add_argument('--style_image',type=str,default='images/style/starry-night.jpg',help='The path of style_image where you store.')
parser.add_argument('--gpu_num',type=int,default=None,help='gpu number if you need')
args = parser.parse_args()

try:
    assert os.path.exists(args.VGG19_Model_Path)
except:
    print ('There is no VGG19_Model file')

try:
    assert os.path.exists(args.content_image)
except:
    print('There is no content_image')

try:
    assert os.path.exists(args.style_image)
except:
    print('There is no style_image')

if args.gpu_num is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = 'args.gpu_num'

content_image = load_image(args.content_image,size=[224,224])
style_image = load_image(args.style_image,size=(content_image.shape[1],content_image.shape[0]))


content_image = np.reshape(content_image, ((1,) + content_image.shape))
style_image = np.reshape(style_image, ((1,) + style_image.shape))
generated_image = generate_noise_image(content_image)





picture_g = tf.Variable(generated_image,trainable=True,dtype=tf.float32)

picture_c = tf.placeholder(tf.float32,shape=content_image.shape,name='content')
picture_s = tf.placeholder(tf.float32,shape=style_image.shape,name='style')

model = VGG19(args.VGG19_Model_Path)



picture = tf.concat([picture_c,picture_s,picture_g],axis=0)
model.build(picture)
out_C = model.conv4_2
a_C, a_S, a_G = tf.split(out_C,[1,1,1],0)

J_content = compute_content_cost(a_C, a_G)

J_style = 0

out_S_1 = model.conv1_1
out_S_2 = model.conv2_1
out_S_3 = model.conv3_1
out_S_4 = model.conv4_1
out_S_5 = model.conv5_1

a_C_1, a_S_1,a_G_1 =tf.split(out_S_1,[1,1,1],axis=0)
J_style_layer_1 = compute_layer_style_cost(a_S_1, a_G_1)
J_style += 0.2 * J_style_layer_1

a_C_2, a_S_2,a_G_2 =tf.split(out_S_2,[1,1,1],axis=0)
J_style_layer_2 = compute_layer_style_cost(a_S_2, a_G_2)
J_style += 0.2 * J_style_layer_2

a_C_3, a_S_3,a_G_3 =tf.split(out_S_3,[1,1,1],axis=0)
J_style_layer_3 = compute_layer_style_cost(a_S_3, a_G_3)
J_style += 0.2 * J_style_layer_3

a_C_4, a_S_4,a_G_4 =tf.split(out_S_4,[1,1,1],axis=0)
J_style_layer_4 = compute_layer_style_cost(a_S_4, a_G_4)
J_style += 0.2 * J_style_layer_4

a_C_5, a_S_5,a_G_5 =tf.split(out_S_5,[1,1,1],axis=0)
J_style_layer_5 = compute_layer_style_cost(a_S_5, a_G_5)
J_style += 0.2 * J_style_layer_5

J_all = total_cost(J_content, J_style, 10, 40)




optimizer = tf.train.AdamOptimizer(2)
train_step = optimizer.minimize(J_all)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(200):

        sess.run(train_step,feed_dict={picture_c: content_image, picture_s: style_image})

        generated_image = sess.run(picture_g)



        if i % 20 == 0:
            J_All, J_Content, J_Style = sess.run([J_all, J_content, J_style],feed_dict={picture_c: content_image, picture_s: style_image})
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(J_All))
            print("content cost = " + str(J_Content))
            print("style cost = " + str(J_Style))

    save_image("images/generated/" + str(args.content_image.split('.')[-2].split('/')[-1])+str('_')+str(args.style_image.split('.')[-2].split('/')[-1]) + ".png",
               undo_preprocess(np.reshape(generated_image,[generated_image.shape[1],generated_image.shape[2],3])))








