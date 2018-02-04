# -*- coding: utf-8 -*-
# @Time    : 2018/2/3 22:28
# @Author  : HLin
# @Email   : linhua2017@ia.ac.cn
# @File    : train.py
# @Software: PyCharm


from utils.utils import *
from models.vgg19_tf import VGG19
import os


os.environ['CUDA_VISIBLE_DEVICES']='0'
content_image = load_image("images/gyeongbokgung.jpg",size=[224,224])
style_image = load_image("images/starry-night.jpg",size=(content_image.shape[1],content_image.shape[0]))


content_image = np.reshape(content_image, ((1,) + content_image.shape))
style_image = np.reshape(style_image, ((1,) + style_image.shape))
generated_image = generate_noise_image(content_image)





picture_g = tf.Variable(generated_image,trainable=True,dtype=tf.float32)

picture_c = tf.placeholder(tf.float32,shape=content_image.shape,name='content')
picture_s = tf.placeholder(tf.float32,shape=style_image.shape,name='style')

model = VGG19("weight/vgg19.npy")



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

J = total_cost(J_content, J_style, 10, 40)




optimizer = tf.train.AdamOptimizer(2)
train_step = optimizer.minimize(J)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(200):

        sess.run(train_step,feed_dict={picture_c: content_image, picture_s: style_image})

        generated_image = sess.run(picture_g)



        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style],feed_dict={picture_c: content_image, picture_s: style_image})
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output/" + str('w')+str(i) + ".png", undo_preprocess(np.reshape(generated_image,[224,224,3])))









