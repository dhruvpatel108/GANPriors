import os
import glob
import utils
import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
from config import argparser
from data_process import noisy_meas
import tensorflow_probability as tfp

tfd= tfp.distributions
tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)

N = PARAMS.n_oed
burn = int(PARAMS.burn_oed*N)
z_dim = PARAMS.z_dim
batch_size = PARAMS.batch_size # should always be one
noise_var = PARAMS.noise_var
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
tile_size = PARAMS.tile_size
iter_no = PARAMS.iter_no

save_dir = './exps/oed/tile{}/img{}_var{}_N{}/iter{}'.format(PARAMS.tile_size, PARAMS.img_no, noise_var, N, iter_no)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)   
    
x_true3d, noisy_mat4d = noisy_meas(batch_size)

        
if (iter_no!=0):
    var_info = np.load(save_dir+'/../var_info.npy')    #np.load('./oed_experiments/oed/tile7x7/digit{}_var{}_N{}/var_info.npy'.format(PARAMS.digit, noise_var, N))
    if (iter_no==1):
        tile_tl_row = int(var_info[iter_no-1,0])
        tile_tl_col = int(var_info[iter_no-1,1])
    else:
        rows = var_info[:iter_no,0].astype(np.int32)
        cols = var_info[:iter_no,1].astype(np.int32)



with tf.Graph().as_default() as g:
    def joint_log_prob(z):         
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = gen_out - tf.constant(noisy_mat4d)
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), scale_diag=np.ones(z_dim, dtype=np.float32))
        dim_window = iter_no*(tile_size**2)*3   #n_channels=3

        if(iter_no==0):
            return prior.log_prob(z)
        elif(iter_no==1):            
            diff_img_visible = tf.reshape(tf.slice(diff_img, [0,tile_tl_row, tile_tl_col, 0], [1,tile_size,tile_size,3]), [int(dim_window/iter_no)])
            like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_window, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_window, dtype=np.float32))
            return (prior.log_prob(z) + like.log_prob(diff_img_visible))
        else:
            ls = []
            for ii in range(iter_no):
                ls.append(tf.reshape(tf.slice(diff_img, [0, rows[ii], cols[ii],0], [1,tile_size,tile_size,3]), [int(dim_window/iter_no)]))
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
    
    variables_to_restore = slim.get_variables_to_restore()   
    saver = tf.train.Saver(variables_to_restore)
    

with tf.Session(graph=g) as sess:
    
    saver.restore(sess, PARAMS.model_path)    
    
    samples_ = sess.run(samples)    
    np.save(save_dir+'/samples.npy', samples_)
    print('acceptance ratio = {}'.format(sess.run(p_accept)))
