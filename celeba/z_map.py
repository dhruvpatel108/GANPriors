# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:25:42 2019

@author: pogo
"""
import glob
import argparse
import utils
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
import matplotlib.pyplot as plt
import matplotlib.patches as patches


tf.reset_default_graph()

batch_size = 1
z_dim = 100
noise_lvl = 0.1
dim_prior = z_dim
prior_stddev = 1.0

seed_no = 1008
np.random.seed(seed_no)
parser = argparse.ArgumentParser()
parser.add_argument('--noise_var', type=float, default=0.1)
parser.add_argument('--img_no', type=int)
PARAMS = parser.parse_args()
print('------------- noise_var = {} ---------------'.format(PARAMS.noise_var))
print('------------- img_no = {}    ---------------'.format(PARAMS.img_no))
''' data '''
def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/test/{}.jpg'.format(PARAMS.img_no))
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)
print(data_pool.batch()[0,:,:,:].shape)

dim_like = 64*64*3
#noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*like_var, size=1)
#np.save('./data/noise3d_var{}_seed{}'.format(like_var, seed_no), np.reshape(noise_mat3d, [64,64,3]))
#noisy_mat3d = data_pool.batch()[0,:,:,:] + np.reshape(noise_mat3d, [64,64,3])
noisy_mat3d = data_pool.batch()[0,:,:,:] + np.load('./data/noise3d_var{}_seed{}.npy'.format(PARAMS.noise_var, seed_no))
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])    
    dummy = tf.Variable(name='dummy', trainable=False, initial_value=z1)   
    assign_op1 = dummy.assign(z1)
    
    gen_out = gen(dummy, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(noisy_mat4d)
    
    loss = 0.5*tf.linalg.norm(diff_img)**2 + (0.5*PARAMS.noise_var*tf.linalg.norm(dummy)**2)
    
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[dummy], options={'maxiter': 10000, 'disp':True})
    
    
    
    
    variables = slim.get_variables_to_restore()    
    variables_to_restore = [v for v in variables if v.name.split(':')[0]!='dummy'] 
    
    saver = tf.train.Saver(variables_to_restore)

z_sample = np.random.normal(size=[batch_size, z_dim])
with tf.Session(graph=g) as sess:
    _ = sess.run(assign_op1, feed_dict={z1:z_sample})  
    
    saver.restore(sess, PARAMS.model_path)
    print('model restored!')
    optimizer.minimize(sess)
    
    g_z_map = sess.run(gen_out)
    
    def normalize(a):
        return((a-np.min(a))/(np.max(a)-np.min(a)))
        
        
    print(np.max(data_pool.batch()[0,:,:,:]))
    print(np.min(data_pool.batch()[0,:,:,:]))
    print('noisy')
    print(np.max(noisy_mat4d[0,:,:,:]))
    print(np.min(noisy_mat4d[0,:,:,:]))
    print('g_z_map')
    print(np.max(g_z_map[0,:,:,:]))
    print(np.min(g_z_map[0,:,:,:]))
    
    
    x_true = normalize(data_pool.batch()[0,:,:,:])
    y_meas = normalize(noisy_mat4d[0,:,:,:])
    x_map = normalize(g_z_map)[0,:,:,:]
    print('x_true - max = {}    |   min = {}'.format(np.max(x_true), np.min(x_true)))
    print('y_meas - max = {}    |   min = {}'.format(np.max(y_meas), np.min(y_meas)))
    print('x_map - max = {}    |   min = {}'.format(np.max(x_map), np.min(x_map)))
    
    
    plt.figure()
    plt.imshow(x_true)
    plt.colorbar()    
    
    plt.figure()
    plt.imshow(y_meas)
    plt.colorbar()
    
    plt.figure()
    plt.imshow(x_map)
    #plt.clim(-1., 1.)
    plt.colorbar()
    plt.show()
    #print(sess.run(loss))
    
    print('Model optimized!')
    
