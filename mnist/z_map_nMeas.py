import os
from config import argparser
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


tf.reset_default_graph()

batch_size = 1
z_dim = 100
noise_lvl = 1.0
tile_a = 49  #7x7
tile_s = int(np.sqrt(tile_a))
img_h = 28
img_w  = 28
img_c = 1
parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int, default=1)
parser.add_argument('--noise_var', type=float, default=noise_lvl)
parser.add_argument('--nMeas', type=int, default=20)
parser.add_argument('--iter_no', type=int)
args = parser.parse_args()
print('------------- digit = {} ---------------'.format(args.digit))
print('------------- noise_var = {} ---------------'.format(args.noise_var))
print('------------- nMeas = {} ---------------'.format(args.nMeas))
print('------------- iter_no = {} ---------------'.format(args.iter_no))
noise_var = args.noise_var
''' data '''
digit_idx=[3,2,1,18,4,8,11,0,61,7]
it_no = 1

test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[digit_idx[args.digit]], [img_h, img_w, img_c])
#np.random.seed(1008)
dim_like = img_h*img_w*img_c
save_dir = './oed_experiments/random_sample_zmap/tile{}x{}/digit{}_var{}'.format(tile_s, tile_s,args.digit, args.noise_var)
ynoisy_file = save_dir+'/noisy_mat4d.npy'

if os.path.exists(ynoisy_file):
    print('|||||||||||||| ********** file for full measurement does exist ***********|||||||||||||||||||||||||')
    noisy_mat4d = np.load(ynoisy_file)
else:
    print('********** file for full measurement does not exist ***********')
    noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1)
    noisy_mat3d = test_sample + np.reshape(noise_mat3d, [img_h, img_w, img_c])
    noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)
    np.save(ynoisy_file, noisy_mat4d)


#indices = np.random.choice(dim_like, size=(args.nMeas,1), replace=False)
ind_mat = np.reshape(np.arange(dim_like), [img_h, img_w]).astype(np.int32)
unq_pixel = 0.
while (unq_pixel!=args.nMeas):
    ind_i = np.random.choice(int(img_h/tile_s), int(args.nMeas/tile_a), replace=True).astype(np.int32)
    ind_j = np.random.choice(int(img_w/tile_s), int(args.nMeas/tile_a), replace=True).astype(np.int32)
    indices = np.zeros((args.nMeas,1)).astype(np.int32)
    k=0
    for kk in range(len(ind_i)):
        indices[tile_a*k:(k+1)*tile_a] = np.reshape(ind_mat[tile_s*ind_i[kk]:tile_s*(ind_i[kk]+1), tile_s*ind_j[kk]:tile_s*(ind_j[kk]+1)], (tile_a,1))
        k+=1    
    mm = np.ones(784)
    mm[indices] = 0.
    unq_pixel = np.sum((mm==0).astype(np.int32))
    print('Unique pixels = {}'.format(unq_pixel))

with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])    
    dummy = tf.Variable(name='dummy', trainable=False, initial_value=z1)   
    assign_op1 = dummy.assign(z1)
    
    gen_out = gen(dummy, reuse=tf.AUTO_REUSE, training=False)
    diff_vec = tf.reshape(gen_out - tf.constant(noisy_mat4d), [dim_like])  
    diff_ = tf.gather_nd(diff_vec, indices)      
    loss = 0.5*tf.linalg.norm(diff_)**2 + (0.5*noise_var*tf.linalg.norm(dummy)**2)
    
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[dummy], options={'maxiter': 10000, 'disp':True})
        
    variables = slim.get_variables_to_restore()    
    variables_to_restore = [v for v in variables if v.name.split(':')[0]!='dummy'] 
    
    model_path = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'    
    saver = tf.train.Saver(variables_to_restore)

z_sample = np.random.normal(size=[batch_size, z_dim])
with tf.Session(graph=g) as sess:
    _ = sess.run(assign_op1, feed_dict={z1:z_sample})  
    
    saver.restore(sess, model_path)
    optimizer.minimize(sess)
    
    g_z_map = sess.run(gen_out)
    rec_loss = (np.linalg.norm(g_z_map[0,:,:,0]-test_sample[:,:,0]))/784   #for tile2x2 we had used L2 norm squared instead of just L2 norm as rec_loss     
    
    print('********** Per-pixel reconstruction loss = {} *************'.format(rec_loss)) 
    mm = np.reshape(mm, (28,28))
    masked_y = np.ma.masked_array(noisy_mat4d[0,:,:,0], mask=mm)
    
    fig, axs = plt.subplots(1,3, figsize=(42,42))
    im1 = axs[0].imshow(masked_y)   
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, ax=axs[0], cax=cax)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    im2 = axs[1].imshow(g_z_map[0,:,:,0])
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, ax=axs[1], cax=cax)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    im3 = axs[2].imshow(test_sample[:,:,0]-g_z_map[0,:,:,0])
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, ax=axs[2], cax=cax)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    #plt.show()
    if args.iter_no==1:
        plt.savefig('{}/digit{}_noisevar{}_nMeas{}_iter{}.png'.format(save_dir, args.digit, args.noise_var, args.nMeas, args.iter_no))   
    
       
    err_file = save_dir + '/rec_err.npy'
    if os.path.exists(err_file):
        rec_err = list(np.load(err_file))
        rec_err.append([args.digit, args.noise_var, args.nMeas, rec_loss])
        np.save(err_file, rec_err)
    else:
        print('********** rec_loss file does not exist ***********')
        rec_err = []
        rec_err.append([args.digit, args.noise_var, args.nMeas, rec_loss])
        np.save(err_file, rec_err)
    
