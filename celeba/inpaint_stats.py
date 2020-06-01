import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
import matplotlib.pyplot as plt
from config import argparser
from data_process import noisy_meas



tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)


N = PARAMS.n_inpaint
burn = int(PARAMS.burn_inpaint*N)
n_eff = N-burn
batch_size = 6400
n_iter = int(n_eff/batch_size)
noise_var = PARAMS.noise_var
z_dim = PARAMS.z_dim
dim_prior = z_dim
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
img_no = PARAMS.img_no

sample_dir = './exps/inpaint/img{}_var{}_N{}_strow{}_endrow{}_stcol{}_endcol{}'.format(PARAMS.img_no, noise_var, N, 
                                                                            PARAMS.start_row, PARAMS.end_row, 
                                                                            PARAMS.start_col, PARAMS.end_col)
mcmc_samps = np.load(sample_dir + '/samples.npy')
eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.hist(eff_samps[:, ii], 50, density=True);
    plt.xlabel(r'z_{} '.format(ii))
plt.tight_layout()
plt.savefig('./{}/hist_eff_samples'.format(sample_dir))
plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.plot(eff_samps[:, ii]);
    plt.ylabel(r'z_{}'.format(ii))
plt.tight_layout()
plt.savefig('./{}/eff_samples'.format(sample_dir))
plt.show()


x_true, noisy_mat4d = noisy_meas(batch_size)

mask = np.ones((batch_size, 64,64,3))
mask[:, PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 0.
dim_inpaint = int(np.sum(mask))
with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(noisy_mat4d)
    visible_img = tf.boolean_mask(diff_img, mask)


    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)

with tf.Session(graph=g) as sess:
    saver.restore(sess, PARAMS.model_path)
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((64,64,3))
    x2_mean = np.zeros((64,64,3))    
    for ii in range(n_iter):
        g_z, diff = sess.run([gen_out, visible_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):
            #loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff[kk,:,:,:])**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2
            loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff)**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2
    
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter    
    var = x2_mean - (x_mean)**2
    map_ind = np.argmin(loss)

    g_z_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})
  
    rec_error = np.linalg.norm(g_z_map[0,:,:,:]-x_true)/dim_like
    
    print(' Max. variance = {} and Min. variance = {}'.format(np.max(var), np.min(var)))
    # normalize each stats between 0 and 1 for plotting with imshow for float dtype
    def normalize(a):
        return((a-np.min(a))/(np.max(a)-np.min(a)))
    

    x_true = normalize(x_true)
    y_meas = normalize(noisy_mat4d[0,:,:,:])
    x_map = normalize(g_z_map)[0,:,:,:]
    x_mean = normalize(x_mean)
    x_var = normalize(var)
    np.save(sample_dir+'/x_true_normalized.npy', x_true)
    np.save(sample_dir+'/y_meas_normalized.npy', y_meas)
    np.save(sample_dir+'/x_mean_normalized.npy', x_mean)
    np.save(sample_dir+'/x_var_normalized.npy', x_var)    
    np.save(sample_dir+'/x_map_normalized.npy', x_map)
   
       
    
    mask_ = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
    mask_[PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 1.
    mx_norm = np.ma.masked_array(y_meas, mask=mask_) 
    mx_norm.data[mx_norm.mask]=1
 
    fig, axs = plt.subplots(3,2, figsize=(20,20))
    im1 = axs[0][0].imshow(x_true)    
    fig.colorbar(im1, ax=axs[0][0])
    axs[0][0].set_title(r'$x_{{true}}$')
    im2 = axs[0][1].imshow(mx_norm)    
    fig.colorbar(im2, ax=axs[0][1])
    axs[0][1].set_title(r'$y_{{meas}}$')
    
    im3 = axs[1][0].imshow(x_map)    
    fig.colorbar(im3, ax=axs[1][0])    
    axs[1][0].set_title(r'$x_{{map}}$')
    im4 = axs[1][1].imshow(x_map-x_true)    
    fig.colorbar(im4, ax=axs[1][1])
    axs[1][1].set_title(r'$x_{{map}} - x_{{true}}$')    
    
    im5 = axs[2][0].imshow(x_mean)    
    fig.colorbar(im5, ax=axs[2][0])
    axs[2][0].set_title(r'$x_{{mean}}$')
    im6 = axs[2][1].imshow(var)    
    fig.colorbar(im6, ax=axs[2][1])
    axs[2][1].set_title(r'$x_{{var}}$')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(sample_dir+'/stats')
    plt.show()