import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
from config import argparser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)

N = PARAMS.n_mcmc #np.size(mcmc_samps, 0)
burn = int(PARAMS.burn_oed*N)
n_eff = N-burn
batch_size = 6400 
n_iter = int(n_eff/batch_size)
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
tile_size = PARAMS.tile_size
noise_var = PARAMS.noise_var
iter_no = PARAMS.iter_no

sample_dir = './exps/oed/tile{}/digit{}_var{}_N{}/iter{}'.format(PARAMS.tile_size, PARAMS.digit, noise_var, N, PARAMS.iter_no)
mcmc_samps = np.load(sample_dir + '/samples.npy')
eff_samps = np.squeeze(mcmc_samps[burn:,:,:])

'''data'''
test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[PARAMS.digit_array[PARAMS.digit]], [PARAMS.img_h, PARAMS.img_w, PARAMS.img_c])
x_true = test_sample[:,:,0]
y_hat4d = np.load(sample_dir+'/y_hat4d.npy')


# histogram of first 25 components of posterior
plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.hist(eff_samps[:, ii], 50, density=True);
    plt.xlabel(r'z_{} '.format(ii))
plt.tight_layout()
plt.savefig('./{}/hist_eff_samples25'.format(sample_dir))
# trace plot of first 25 components of posterior
plt.figure(figsize=(15, 6))
for ii in range(25):
    plt.subplot(5,5,ii+1)
    plt.plot(eff_samps[:, ii]);
    plt.ylabel(r'z_{}'.format(ii))
plt.tight_layout()
plt.savefig('./{}/eff_samples25'.format(sample_dir))
plt.show()


# tile info
if (iter_no!=0):
    var_info = np.load(sample_dir+'/../var_info.npy'.format(sample_dir))
    #var_info = np.load('./exps/oed/tile{}/digit{}_var{}_N{}/var_info.npy'.format(PARAMS.tile_size, PARAMS.digit, noise_var, N))
    if (iter_no==1):
        tile_tl_row = int(var_info[iter_no-1,0])
        tile_tl_col = int(var_info[iter_no-1,1])
    else:
        rows = var_info[:iter_no,0].astype(np.int32)
        cols = var_info[:iter_no,1].astype(np.int32)


with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, PARAMS.z_dim])         
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(y_hat4d)

    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)



with tf.Session(graph=g) as sess:
    saver.restore(sess, PARAMS.model_path)
    
    loss = np.zeros((n_eff))
    x_mean = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
    x2_mean = np.zeros((PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))    
    for ii in range(n_iter):
        g_z, diff = sess.run([gen_out, diff_img], feed_dict={z1:eff_samps[ii*batch_size:(ii+1)*batch_size, :]})
        x_mean = x_mean + np.mean(g_z, axis=0)
        x2_mean = x2_mean + np.mean(g_z**2, axis=0)
        for kk in range(batch_size):
            if (iter_no==0):
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2)
            elif (iter_no==1):
                diff_vec = np.reshape(diff[kk, tile_tl_row:tile_tl_row+tile_size, tile_tl_col:tile_tl_col+tile_size, :], tile_size**2)
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_vec)**2
            else:
                ls = []
                for hh in range(iter_no):
                    ls.append(np.reshape(diff[kk, rows[hh]:rows[hh]+tile_size, cols[hh]:cols[hh]+tile_size, :], tile_size**2))
                diff_vec = np.concatenate(ls) 
                loss[(ii*batch_size)+kk] = (0.5*noise_var*np.linalg.norm(eff_samps[(ii*batch_size)+kk, :])**2) + 0.5*np.linalg.norm(diff_vec)**2
            #diff_vec = np.reshape(diff[kk,7:14,14:21,:], tile_size**2)
            #diff_vec = np.concatenate((np.reshape(diff[kk,7:14,14:21,:], tile_size**2), np.reshape(diff[kk,14:21,7:14,:], tile_size**2), np.reshape(diff[kk,14:21,14:21,:], tile_size**2)))
            

    
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
    print('Avg Block variance is:')
    print(block_var)
    row_ind, col_ind = np.unravel_index(np.argmax(block_var), block_var.shape)
    print('row_ind={} and col_ind={} for max. var'.format(row_ind, col_ind))
    
    mask_ = np.ones((1, PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
    if (iter_no==0):
        var_info = np.zeros((100,4))
        var_info[iter_no,0] = (row_ind*tile_size)
        var_info[iter_no,1] = (col_ind*tile_size)
        var_info[iter_no,2] = block_var[row_ind, col_ind]
        rec_loss = np.linalg.norm(x_map[0,:,:,0]-test_sample[:,:,0])/784
        var_info[iter_no,3] = rec_loss
        np.save(sample_dir+'/../var_info.npy',var_info)
        #mask_[:, int(var_info[iter_no,0]):int(var_info[iter_no,0])+tile_size, int(var_info[iter_no,1]):int(var_info[iter_no,1])+tile_size, :] = 0
    else:
        var_info[iter_no,0] = (row_ind*tile_size)
        var_info[iter_no,1] = (col_ind*tile_size)
        var_info[iter_no,2] = block_var[row_ind, col_ind]
        rec_loss = np.linalg.norm(x_map[0,:,:,0]-test_sample[:,:,0])/784
        var_info[iter_no,3] = rec_loss
        np.save(sample_dir+'/../var_info.npy', var_info)
        for ii in range(iter_no):
            mask_[:, int(var_info[ii,0]):int(var_info[ii,0])+tile_size, int(var_info[ii,1]):int(var_info[ii,1])+tile_size, :] = 0.
        
    mx = np.ma.masked_array(np.expand_dims(y_hat4d[0,:,:,:], axis=0), mask=mask_)
    print('iter{}, rec_loss={}'.format(PARAMS.iter_no, rec_loss))      
    
    fig, axs = plt.subplots(3,2)  
    im1 = axs[0][0].imshow(x_true)   
    divider = make_axes_locatable(axs[0][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, ax=axs[0][0], cax=cax)
    axs[0][0].set_title(r'$x_{{true}}$')
    axs[0][0].set_xticks([])
    axs[0][0].set_yticks([])
    
    im2 = axs[0][1].imshow(mx[0,:,:,0])   
    divider = make_axes_locatable(axs[0][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, ax=axs[0][1], cax=cax)
    axs[0][1].set_title(r'$y_{{meas}}$')
    axs[0][1].set_xticks([])
    axs[0][1].set_yticks([])
    
    im3 = axs[1][0].imshow(x_map[0,:,:,0])   
    divider = make_axes_locatable(axs[1][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, ax=axs[1][0], cax=cax)
    axs[1][0].set_title(r'$x_{{map}}$')
    axs[1][0].set_xticks([])
    axs[1][0].set_yticks([])
    
    im4 = axs[1][1].imshow(x_map[0,:,:,0]-x_true)   
    divider = make_axes_locatable(axs[1][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4, ax=axs[1][1], cax=cax)
    axs[1][1].set_title(r'$x_{{map}} - x_{{true}}$')
    axs[1][1].set_xticks([])
    axs[1][1].set_yticks([])    
    
    im5 = axs[2][0].imshow(x_mean[:,:,0])   
    divider = make_axes_locatable(axs[2][0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im5, ax=axs[2][0], cax=cax)
    axs[2][0].set_title(r'$x_{{mean}}$')
    axs[2][0].set_xticks([])
    axs[2][0].set_yticks([])    
    
    im6 = axs[2][1].imshow(var[:,:,0])   
    divider = make_axes_locatable(axs[2][1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im6, ax=axs[2][1], cax=cax)
    axs[2][1].set_title(r'$x_{{var}}$')
    axs[2][1].set_xticks([])
    axs[2][1].set_yticks([])
    axs[2][1].add_patch(patches.Rectangle((var_info[iter_no,1],var_info[iter_no,0]),(tile_size),(tile_size),linewidth=1,edgecolor='r',facecolor='none'))
    plt.savefig('./{}/stats_iter{}'.format(sample_dir, iter_no))    
    
    np.save(sample_dir+'/x_true', x_true)
    np.save(sample_dir+'/x_map', x_map[0,:,:,0])
    np.save(sample_dir+'/x_mean', x_mean[:,:,0])
    np.save(sample_dir+'/x_var', var[:,:,0])        
    np.save(sample_dir+'/mask', mask_) 
    plt.show()    
