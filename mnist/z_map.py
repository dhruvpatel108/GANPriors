import os
from config import argparser
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import matplotlib.pyplot as plt

tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c


test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[PARAMS.digit_array[PARAMS.digit]], [PARAMS.img_h, PARAMS.img_w, PARAMS.img_c])

if not os.path.exists(meas_path):
    print('***** Measurement does not exist !! *****')
    print('** Generating one with noise_var={} **'.format(PARAMS.noise_var))
    os.makedirs(meas_path)
    noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*PARAMS.noise_var, size=1)    
    np.save(meas_path+'/noise_mat3d.npy', noise_mat3d)    
else:
    noise_mat3d = np.load(meas_path+'/noise_mat3d.npy')
    
    
y_hat3d = test_sample + np.reshape(noise_mat3d, [PARAMS.img_h, PARAMS.img_w, PARAMS.img_c])
y_hat4d = np.tile(y_hat3d, (PARAMS.batch_size, 1, 1, 1)).astype(np.float32)
    
    

with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[PARAMS.batch_size, PARAMS.z_dim])    
    dummy = tf.Variable(name='dummy', trainable=False, initial_value=z1)   
    assign_op1 = dummy.assign(z1)
    
    gen_out = gen(dummy, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(y_hat4d)
    
    loss = 0.5*tf.linalg.norm(diff_img)**2 + (0.5*PARAMS.noise_var*tf.linalg.norm(dummy)**2)    
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[dummy], options={'maxiter': 10000, 'disp':True})    
    
    variables = slim.get_variables_to_restore()    
    variables_to_restore = [v for v in variables if v.name.split(':')[0]!='dummy']     
    saver = tf.train.Saver(variables_to_restore)


z_sample = np.random.normal(size=[PARAMS.batch_size, PARAMS.z_dim])
with tf.Session(graph=g) as sess:
    _ = sess.run(assign_op1, feed_dict={z1:z_sample})  
    
    saver.restore(sess, PARAMS.model_path)
    optimizer.minimize(sess)
    
    g_z_map = sess.run(gen_out)
    
    
    plt.figure()
    plt.imshow(test_sample[:,:,0])
    plt.colorbar()    
    
    plt.figure()
    plt.imshow(y_hat4d[0,:,:,0])
    plt.colorbar()
    
    plt.figure()
    plt.imshow(g_z_map[0,:,:,0])
    plt.clim(-1., 1.)
    plt.colorbar()
    
    plt.figure()
    plt.imshow(test_sample[:,:,0]-g_z_map[0,:,:,0])
    plt.colorbar()
    plt.title(r'g($z_{{map}}$) - x | Reconstruction error(per pixel)={}'.format((np.linalg.norm(g_z_map[0,:,:,0]-test_sample[:,:,0])**2)/dim_like)) 
    plt.show()
    
    print('Model restored!')
