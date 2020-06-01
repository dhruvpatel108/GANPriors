import glob
import utils
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import argparser
from data_process import noisy_meas


tf.reset_default_graph()
PARAMS = argparser()

N = PARAMS.n_oed
burn = int(PARAMS.burn_oed*N)
n_eff = N-burn
z_dim = PARAMS.z_dim
batch_size = 6400
n_iter = int(n_eff/batch_size)
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
tile_size = PARAMS.tile_size
noise_var = PARAMS.noise_var
n_chnls = PARAMS.img_c
iter_no = PARAMS.iter_no


sample_dir = './exps/oed/tile{}/img{}_var{}_N{}/iter{}'.format(PARAMS.tile_size, PARAMS.img_no, noise_var, N, iter_no)
mcmc_samps = np.load(sample_dir + '/samples.npy')
eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

x_true, noisy_mat4d = noisy_meas(batch_size)

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
    var_info = np.load(sample_dir+'/../var_info.npy'.format(sample_dir))
    #var_info = np.load('./oed_experiments/oed/tile2x2/digit{}_var{}_N{}/var_info.npy'.format(PARAMS.digit, noise_var, N))
    if (iter_no==1):
        tile_tl_row = int(var_info[iter_no-1,0])
        tile_tl_col = int(var_info[iter_no-1,1])
    else:
        rows = var_info[:iter_no,0].astype(np.int32)
        cols = var_info[:iter_no,1].astype(np.int32)


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(noisy_mat4d)

    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)

with tf.Session(graph=g) as sess:
    saver.restore(sess, PARAMS.model_path)
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((64,64,3))
    x2_mean = np.zeros((64,64,3))    
    for ii in range(n_iter):
        g_z, diff = sess.run([gen_out, diff_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):
            if (iter_no==0):
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2)
            elif (iter_no==1):
                diff_vec = np.reshape(diff[kk, tile_tl_row:tile_tl_row+tile_size, tile_tl_col:tile_tl_col+tile_size, :], (tile_size**2)*n_chnls)
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_vec)**2
            else:
                ls = []
                for hh in range(iter_no):
                    ls.append(np.reshape(diff[kk, rows[hh]:rows[hh]+tile_size, cols[hh]:cols[hh]+tile_size, :], (tile_size**2)*n_chnls))
                diff_vec = np.concatenate(ls) 
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_vec)**2
            #diff_vec = np.reshape(diff[kk,7:14,14:21,:], tile_size**2)
            #diff_vec = np.concatenate((np.reshape(diff[kk,7:14,14:21,:], tile_size**2), np.reshape(diff[kk,14:21,7:14,:], tile_size**2), np.reshape(diff[kk,14:21,14:21,:], tile_size**2)))
            

    
    x_mean = x_mean/n_iter
    x2_mean = x2_mean/n_iter    
    var = x2_mean - (x_mean)**2
    map_ind = np.argmin(loss)

    g_z_map = sess.run(gen_out, feed_dict={z1: np.tile(np.expand_dims(eff_samps[map_ind,:], axis=0), (batch_size, 1))})
    rec_loss = np.linalg.norm(g_z_map[0,:,:,:]-x_true)/dim_like

    n_tiles = int(64/tile_size)
    block_var = np.zeros((n_tiles, n_tiles))  
    for i in range(n_tiles):
        for j in range(n_tiles):
            block_var[i,j] = np.mean(var[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size])
    print('Avg Block variance is:')
    print(block_var)
    row_ind, col_ind = np.unravel_index(np.argmax(block_var), block_var.shape)
    print('row_ind={} and col_ind={} for max. var'.format(row_ind, col_ind))
    
    mask_ = np.ones((64,64,3))
    if (iter_no==0):
        var_info = np.zeros((100,4))
        var_info[iter_no,0] = (row_ind*tile_size)
        var_info[iter_no,1] = (col_ind*tile_size)
        var_info[iter_no,2] = block_var[row_ind, col_ind]
        var_info[iter_no,3] = rec_loss
        np.save(sample_dir+'/../var_info.npy',var_info)
        #mask_[:, int(var_info[iter_no,0]):int(var_info[iter_no,0])+tile_size, int(var_info[iter_no,1]):int(var_info[iter_no,1])+tile_size, :] = 0
    else:
        var_info[iter_no,0] = (row_ind*tile_size)
        var_info[iter_no,1] = (col_ind*tile_size)
        var_info[iter_no,2] = block_var[row_ind, col_ind]
        var_info[iter_no,3] = rec_loss
        np.save(sample_dir+'/../var_info.npy', var_info)
        for ii in range(iter_no):
            mask_[int(var_info[ii,0]):int(var_info[ii,0])+tile_size, int(var_info[ii,1]):int(var_info[ii,1])+tile_size, :] = 0.
        
    mx = np.ma.masked_array(noisy_mat4d[0,:,:,:], mask=mask_)
    
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
    np.save(sample_dir+'/mask_.npy', mask_)
    
    mx_normalized = np.ma.masked_array(y_meas, mask=mask_) 
    mx_normalized.data[mx_normalized.mask]=1
    #mx_normalized.data[mx_normalized.mask]=1
    
    
    fig, axs = plt.subplots(3,2)  
    im1 = axs[0][0].imshow(x_true)   
    #divider = make_axes_locatable(axs[0][0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im1, ax=axs[0][0], cax=cax)
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    
    im2 = axs[0][1].imshow(mx_normalized)   
    #divider = make_axes_locatable(axs[0][1])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im2, ax=axs[0][1], cax=cax)
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])
    
    im3 = axs[1][0].imshow(x_map)   
    #divider = make_axes_locatable(axs[1][0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im3, ax=axs[1][0], cax=cax)
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    
    im4 = axs[1][1].imshow(x_map - x_true)   
    #divider = make_axes_locatable(axs[1][1])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im4, ax=axs[1][1], cax=cax)
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])    
    #axs[1,1].set_title(rec_loss)
    
    im5 = axs[2][0].imshow(x_mean)   
    #divider = make_axes_locatable(axs[2][0])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im5, ax=axs[2][0], cax=cax)
    axs[2][0].set_xticks([])
    axs[2][0].set_yticks([])    
    
    im6 = axs[2][1].imshow(x_var)   
    #divider = make_axes_locatable(axs[2][1])
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #fig.colorbar(im6, ax=axs[2][1], cax=cax)
    #axs[2,1].set_title(np.mean(x_var))
    axs[2][1].set_xticks([])
    axs[2][1].set_yticks([])
    axs[2][1].add_patch(patches.Rectangle((var_info[iter_no,1],var_info[iter_no,0]),(tile_size),(tile_size),linewidth=1,edgecolor='r',facecolor='none'))
    plt.savefig('./{}/stats_iter{}'.format(sample_dir, iter_no))    
    
    plt.show()    
    
