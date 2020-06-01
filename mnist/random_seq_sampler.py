import os
import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import data_mnist as data
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd= tfp.distributions

tf.reset_default_graph()
N = 64000
burn = int(0.5*N)
z_dim = 100
batch_size = 1 # should always be one
noise_lvl = 1.0
dim_like = 28*28*1
tile_size = 7
seed_no = 1008
np.random.seed(seed_no)


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
if (PARAMS.iter_no==0):
    print('**** noise file does not exist... creating new one ****')
    noise_mat3d = np.reshape(np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1), (28,28,1))
    np.save(noisefile_path, noise_mat3d)
noisy_mat3d = np.load(noisefile_path) + test_sample
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)    
#noisy_mat4d = np.load('./oed_experiments/random_sample_zmap/tile2x2/digit{}_var{}/noisy_mat4d.npy'.format(PARAMS.digit, PARAMS.noise_var))
save_dir = './oed_experiments/random_sequential/tile7x7/digit{}_var{}_N{}/iter{}'.format(PARAMS.digit, noise_var, N, iter_no)
if not os.path.exists(save_dir):
        os.makedirs(save_dir)   
if(iter_no==1):
    window_info = np.zeros((100,4))
    ind_i = np.random.choice(4,1)
    ind_j = np.random.choice(4,1)
    tile_tl_row = int(ind_i*tile_size)
    tile_tl_col = int(ind_j*tile_size)
    window_info[iter_no-1,0] = tile_tl_row
    window_info[iter_no-1,1] = tile_tl_col
    np.save(save_dir+'/../window_info.npy', window_info)
else:
    w_info = np.load(save_dir+'/../window_info.npy')
    ind_i = 1
    ind_j = 2
    #ind_i = np.random.choice(4,1)
    #ind_j = np.random.choice(4,1)
    #while((ind_i==w_info[:iter_no-1, 0].any())and(ind_j==w_info[:iter_no-1, 1].any())):
    #    ind_i = np.random.choice(4,1)
    #    ind_j = np.random.choice(4,1)
    tile_tl_row = int(ind_i*tile_size)
    tile_tl_col = int(ind_j*tile_size)
    w_info[iter_no-1,0] = tile_tl_row
    w_info[iter_no-1,1] = tile_tl_col
    print('***** window locations till now *******')
    print(w_info[:iter_no,:])
    np.save(save_dir+'/../window_info.npy', w_info)
    rows = w_info[:iter_no,0].astype(np.int32)
    cols = w_info[:iter_no,1].astype(np.int32)



with tf.Graph().as_default() as g:
    def joint_log_prob(z):         
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = gen_out - tf.constant(noisy_mat4d)
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), scale_diag=np.ones(z_dim, dtype=np.float32))
        dim_window = iter_no*(tile_size**2)

        if(iter_no==1):            
            diff_img_visible = tf.reshape(tf.slice(diff_img, [0,tile_tl_row, tile_tl_col, 0], [1,tile_size,tile_size,1]), [int(dim_window/iter_no)])
            like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_window, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_window, dtype=np.float32))
            return (prior.log_prob(z) + like.log_prob(diff_img_visible))
        else:
            ls = []
            for ii in range(iter_no):
                ls.append(tf.reshape(tf.slice(diff_img, [0, rows[ii], cols[ii],0], [1,tile_size,tile_size,1]), [int(dim_window/iter_no)]))
            diff_img_visible = tf.concat(ls, axis=0)
            like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_window, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_window, dtype=np.float32))        
            return (prior.log_prob(z) + like.log_prob(diff_img_visible))
                                          

    def unnormalized_posterior(z):
        return joint_log_prob(z) 
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                    tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_posterior, step_size=np.float32(0.1), num_leapfrog_steps=3),
                   num_adaptation_steps=int(0.8*burn))    
    

    initial_state = tf.constant(np.random.normal(size=[batch_size, z_dim]).astype(np.float32))
    samples, [st_size, log_accept_ratio] = tfp.mcmc.sample_chain(
      num_results=N,
      num_burnin_steps=burn,
      current_state=initial_state,
      kernel=adaptive_hmc,
      trace_fn=lambda _, pkr: [pkr.inner_results.accepted_results.step_size,
                             pkr.inner_results.log_accept_ratio])
    p_accept = tf.reduce_mean(tf.exp(tf.minimum(log_accept_ratio, 0.)))

     
    zz = tf.placeholder(tf.float32, shape=[N-burn, z_dim])    
    gen_out1 = gen(zz, reuse=tf.AUTO_REUSE, training=False)
    
    model_path = './checkpoints/mnist_wgan_gp1000/Epoch_(999)_(171of171).ckpt'    
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)
    

with tf.Session(graph=g) as sess:
    
    saver.restore(sess, model_path)    
        
    samples_ = sess.run(samples) 
       
    np.save(save_dir+'/samples.npy', samples_)
    #np.save(save_dir+'/noisy_mat4d.npy', noisy_mat4d)
    #print('acceptance ratio = {}'.format(sess.run(p_accept)))
