import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import data_mnist as data
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches

tile_size = 7
iter_no = 3
digit = 1
noise_var = 1.0
n_meas = 20
sample_dir = 'mcmc/random_sampling/digit{}_var{}_nMeas{}_N64000/'.format(digit,noise_var,n_meas)
mcmc_samps = np.load(sample_dir + 'samples.npy')

burn = 32000
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

seed_no = 1008
N = np.size(mcmc_samps, 0)
burn = int(0.5*N)
n_eff = N-burn
batch_size = 640
z_dim = 100
n_iter = int(n_eff/batch_size)
dim_like = 28*28*1

''' data '''
parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int, default=digit)
digit_in = parser.parse_args()
print('------------- digit_in = {} ---------------'.format(digit_in.digit))

digit_idx=[3,2,1,18,4,8,11,0,61,7]

test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[digit_idx[digit_in.digit]], [28,28,1])
np.random.seed(1008)
#noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1)
#noisy_mat3d = test_sample + np.reshape(noise_mat3d, [28,28,1])
#noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)
noisy_mat4d = np.load(sample_dir+'noisy_mat4d.npy')
y_mask = np.load(sample_dir+'/mask_y.npy')
indices = np.load(sample_dir+'/indices.npy')

with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_vec = tf.reshape(gen_out - tf.constant(noisy_mat4d), [dim_like])
    diff_nodal = tf.gather_nd(diff_vec, indices)


    model_path = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'    
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)

with tf.Session(graph=g) as sess:
    saver.restore(sess, model_path)
    #eff_samps = np.squeeze(mcmc_samps[burn:,:,:])
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((28,28,1))
    x2_mean = np.zeros((28,28,1))    
    for ii in range(n_iter):
        g_z, diff_ = sess.run([gen_out, diff_nodal], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):      
        
            loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_)**2
    
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter    
    var = x2_mean - (x_mean)**2
    map_ind = np.argmin(loss)

    x_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})

    n_tiles = int(28/tile_size)
    block_var = np.zeros((n_tiles, n_tiles))  
    for i in range(n_tiles):
        for j in range(n_tiles):
            block_var[i,j] = np.mean(var[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size])
    print('Avg Vlock variance is:')
    print(block_var)
    row_ind, col_ind = np.unravel_index(np.argmax(block_var), block_var.shape)
    print('row_ind={} and col_ind={} for max. var'.format(row_ind, col_ind))
    
    fig, axs = plt.subplots(3,2)
    intervals = float(7)

    im1 = axs[0][0].imshow(test_sample[:,:,0])    
    fig.colorbar(im1, ax=axs[0][0])
    axs[0][0].set_title('x')
    im2 = axs[0][1].imshow(noisy_mat4d[0,:,:,0])    
    fig.colorbar(im2, ax=axs[0][1])
    axs[0][1].set_title('y')
    
    im3 = axs[1][0].imshow(x_map[0,:,:,0])    
    fig.colorbar(im3, ax=axs[1][0])    
    axs[1][0].set_title('g(z_map)')
    im4 = axs[1][1].imshow(x_map[0,:,:,0]-test_sample[:,:,0])    
    fig.colorbar(im4, ax=axs[1][1])
    axs[1][1].set_title('g(z_map) - x | Reconstruction error(per pixel)={}'.format((np.linalg.norm(x_map[0,:,:,0]-test_sample[:,:,0])**2)/784))    
    
    im5 = axs[2][0].imshow(x_mean[:,:,0])    
    fig.colorbar(im5, ax=axs[2][0])
    axs[2][0].set_title('x_mean')
    im6 = axs[2][1].imshow(var[:,:,0])    
    fig.colorbar(im6, ax=axs[2][1])
    axs[2][1].set_title('var - x_TL={}, y_TL={}'.format(col_ind*tile_size, row_ind*tile_size))
    axs[2][1].add_patch(patches.Rectangle((col_ind*tile_size,row_ind*tile_size),(tile_size),(tile_size),linewidth=1,edgecolor='r',facecolor='none'))
    #plt.grid(True, color='w')
    loc = plticker.MultipleLocator(base=intervals)
    axs[2][1].xaxis.set_major_locator(loc)
    axs[2][1].yaxis.set_major_locator(loc)
    #plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    #plt.savefig('./{}/stats_iter{}'.format(sample_dir, iter_no))
    plt.show()
