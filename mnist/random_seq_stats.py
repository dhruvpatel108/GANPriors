import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


tf.reset_default_graph()
N = 64000
burn = int(0.5*N)
n_eff = N-burn
z_dim = 100
batch_size = 6400
n_iter = int(n_eff/batch_size)
dim_like = 28*28*1
tile_size = 7
noise_lvl = 1.0
seed_no = 1008

parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int, default=1)
parser.add_argument('--noise_var', type=float, default=noise_lvl)
parser.add_argument('--iter_no', type=int)
PARAMS = parser.parse_args()
print('------------- digit_in = {} ---------------'.format(PARAMS.digit))
print('------------- noise_var = {} ---------------'.format(PARAMS.noise_var))
print('------------- iter_no = {} ---------------'.format(PARAMS.iter_no))

noise_var = PARAMS.noise_var
iter_no = PARAMS.iter_no
''' data '''
nice_digit_idx = [10,2,1,32,4,15,11,0,128,12]
test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[nice_digit_idx[PARAMS.digit]], [28,28,1])
noisefile_path='./oed_experiments/oed/tile7x7/digit{}_var{}_N{}/noise3d_var{}_seed{}.npy'.format(PARAMS.digit, noise_var, N, noise_var, seed_no)
noisy_mat3d = np.load(noisefile_path) + test_sample
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)  
#noisy_mat4d = np.load('./oed_experiments/random_sample_zmap/tile2x2/digit{}_var{}/noisy_mat4d.npy'.format(PARAMS.digit, PARAMS.noise_var))


sample_dir = './oed_experiments/random_sequential/tile7x7/digit{}_var{}_N{}/iter{}'.format(PARAMS.digit, noise_var, N, iter_no)
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
#plt.show()


if (iter_no!=0):
    w_info = np.load(sample_dir+'/../window_info.npy'.format(sample_dir))
    if (iter_no==1):
        tile_tl_row = int(w_info[iter_no-1,0])
        tile_tl_col = int(w_info[iter_no-1,1])
    else:
        rows = w_info[:iter_no,0].astype(np.int32)
        cols = w_info[:iter_no,1].astype(np.int32)


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(noisy_mat4d)

    model_path = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'    
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)

with tf.Session(graph=g) as sess:
    saver.restore(sess, model_path)
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((28,28,1))
    x2_mean = np.zeros((28,28,1))    
    for ii in range(n_iter):
        g_z, diff = sess.run([gen_out, diff_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):
            # NOTE: for iter_no==0, the results will be same as 0th iteration of oed for same seed and hence ignored.
            if (iter_no==1):
                diff_vec = np.reshape(diff[kk, tile_tl_row:tile_tl_row+tile_size, tile_tl_col:tile_tl_col+tile_size, :], tile_size**2)
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_vec)**2
            else:
                ls = []
                for hh in range(iter_no):
                    ls.append(np.reshape(diff[kk, rows[hh]:rows[hh]+tile_size, cols[hh]:cols[hh]+tile_size, :], tile_size**2))
                diff_vec = np.concatenate(ls) 
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_vec)**2
            

    
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter    
    var = x2_mean - (x_mean)**2
    map_ind = np.argmin(loss)

    x_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})
    rec_loss = np.linalg.norm(x_map[0,:,:,0]-test_sample[:,:,0])/784
    w_info[iter_no-1, 2] = rec_loss
    np.save(sample_dir+'/../window_info.npy', w_info)
    
    
    mask_ = np.ones((1,28,28,1))
    if (iter_no>0):
       for ii in range(iter_no):
            mask_[:, int(w_info[ii,0]):int(w_info[ii,0])+tile_size, int(w_info[ii,1]):int(w_info[ii,1])+tile_size, :] = 0.      
    mx = np.ma.masked_array(np.expand_dims(noisy_mat4d[0,:,:,:], axis=0), mask=mask_)
    
    fig, axs = plt.subplots(3,2)  
    im1 = axs[0][0].imshow(test_sample[:,:,0])   
    divider = make_axes_locatable(axs[0][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, ax=axs[0][0], cax=cax)
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    
    im2 = axs[0][1].imshow(mx[0,:,:,0])   
    divider = make_axes_locatable(axs[0][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, ax=axs[0][1], cax=cax)
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])
    
    im3 = axs[1][0].imshow(x_map[0,:,:,0])   
    divider = make_axes_locatable(axs[1][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, ax=axs[1][0], cax=cax)
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    
    im4 = axs[1][1].imshow(x_map[0,:,:,0]-test_sample[:,:,0])   
    divider = make_axes_locatable(axs[1][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4, ax=axs[1][1], cax=cax)
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])    
    
    im5 = axs[2][0].imshow(x_mean[:,:,0])   
    divider = make_axes_locatable(axs[2][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im5, ax=axs[2][0], cax=cax)
    axs[2][0].set_xticks([])
    axs[2][0].set_yticks([])    
    
    im6 = axs[2][1].imshow(var[:,:,0])   
    divider = make_axes_locatable(axs[2][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im6, ax=axs[2][1], cax=cax)
    axs[2][1].set_xticks([])
    axs[2][1].set_yticks([])
    #axs[2][1].add_patch(patches.Rectangle((w_info[iter_no,1],w_info[iter_no,0]),(tile_size),(tile_size),linewidth=1,edgecolor='r',facecolor='none'))
    plt.savefig('./{}/stats_iter{}'.format(sample_dir, iter_no))    
    
    np.save(sample_dir+'/x_true', test_sample[:,:,0])
    np.save(sample_dir+'/x_map', x_map[0,:,:,0])
    np.save(sample_dir+'/x_mean', x_mean[:,:,0])
    np.save(sample_dir+'/x_var', var[:,:,0])    
    np.save(sample_dir+'/mask', mask_)
    #plt.show()    
