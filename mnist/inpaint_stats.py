import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import argparser
from models_mnist import generator as gen
import matplotlib.pyplot as plt

tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)

N = PARAMS.n_inpaint  #np.size(mcmc_samps, 0)
burn = int(PARAMS.burn_inpaint*N)
n_eff = N-burn
batch_size = 6400
z_dim = PARAMS.z_dim
noise_var = PARAMS.noise_var
n_iter = int(n_eff/batch_size)
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c

sample_dir = './exps/inpaint_exps/digit{}_var{}_N{}_strow{}_endrow{}_stcol{}_endcol{}'.format(PARAMS.digit, noise_var, N, 
                                                                                PARAMS.start_row, PARAMS.end_row, PARAMS.start_col, PARAMS.end_col)
mcmc_samps = np.load(sample_dir + '/samples.npy')
eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

'''data'''
test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[PARAMS.digit_array[PARAMS.digit]], [PARAMS.img_h, PARAMS.img_w, PARAMS.img_c])
x_true = test_sample[:,:,0]
y_hat4d = np.load(sample_dir+'/y_hat4d.npy')

# inpaint mask
mask = np.ones((batch_size, PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
mask[:, PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 0.
dim_inpaint = int(np.sum(mask))

# histogram of first 25 components of posterior
plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.hist(eff_samps[:, ii], 50, density=True);
    plt.xlabel('z_{}'.format(ii))
plt.tight_layout()
plt.savefig('./{}/hist_eff_samples25'.format(sample_dir))
# trace plot of first 25 components of posterior
plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.plot(mcmc_samps[:, 0, ii])
    plt.ylabel('z_{}'.format(ii))
plt.tight_layout()
plt.savefig('./{}/eff_samples25'.format(sample_dir))
plt.show()


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(y_hat4d)
    visible_img = tf.boolean_mask(diff_img, mask)
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)

with tf.Session(graph=g) as sess:
    saver.restore(sess, PARAMS.model_path)    
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
    x2_mean = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))    
    for ii in range(n_iter):
        g_z, diff = sess.run([gen_out, visible_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):
            loss[(ii*batch_size)+kk] = 0.5*np.linalg.norm(diff)**2 + 0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2

    
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter    
    var = x2_mean - (x_mean)**2
    map_ind = np.argmin(loss)
    x_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})
 
    rec_error = np.linalg.norm(x_map[0,:,:,0]-x_true)/dim_like
   
    mask_ = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
    mask_[PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 1.
    mx = np.ma.masked_array(y_hat4d[0,:,:,:],mask=mask_) 
 
    fig, axs = plt.subplots(3,2, figsize=(20,20))
    im1 = axs[0][0].imshow(x_true)    
    fig.colorbar(im1, ax=axs[0][0])
    axs[0][0].set_title(r'$x_{{true}}$')
    im2 = axs[0][1].imshow(mx[:,:,0])    
    fig.colorbar(im2, ax=axs[0][1])
    axs[0][1].set_title(r'$y_{{meas}}$')
    
    im3 = axs[1][0].imshow(x_map[0,:,:,0])    
    fig.colorbar(im3, ax=axs[1][0])    
    axs[1][0].set_title(r'$x_{{map}}$')
    im4 = axs[1][1].imshow(x_map[0,:,:,0]-x_true)    
    fig.colorbar(im4, ax=axs[1][1])
    axs[1][1].set_title(r'$x_{{map}} - x_{{true}}$')    
    
    im5 = axs[2][0].imshow(x_mean[:,:,0])    
    fig.colorbar(im5, ax=axs[2][0])
    axs[2][0].set_title(r'$x_{{mean}}$')
    im6 = axs[2][1].imshow(var[:,:,0])    
    fig.colorbar(im6, ax=axs[2][1])
    axs[2][1].set_title(r'$x_{{var}}$')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(sample_dir+'/stats')
    
    np.save(sample_dir+'/x_true.npy', x_true)
    np.save(sample_dir+'/x_var.npy', var)
    np.save(sample_dir+'/x_mean.npy', x_mean)
    np.save(sample_dir+'/x_map.npy', x_map)
    np.save(sample_dir+'/mask_.npy', mask_)
    plt.show()
    
