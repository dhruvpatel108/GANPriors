import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
from config import argparser
import tensorflow_probability as tfp
tfd= tfp.distributions

tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)

# oed parameters
N = PARAMS.n_oed
burn = int(PARAMS.burn_oed*N)
z_dim = PARAMS.z_dim
batch_size = PARAMS.batch_size # should always be one
noise_var = PARAMS.noise_var
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
tile_size = PARAMS.tile_size


'''data'''
test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[PARAMS.digit_array[PARAMS.digit]], [PARAMS.img_h, PARAMS.img_w, PARAMS.img_c])
meas_path = './meas/full_gaussian_noisevar{}'.format(PARAMS.noise_var)
save_dir = './exps/oed/tile{}/digit{}_var{}_N{}/iter{}'.format(PARAMS.tile_size, PARAMS.digit, noise_var, N, PARAMS.iter_no)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  
    
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
np.save(save_dir+'/y_hat4d.npy', y_hat4d)


# tile location info
if (PARAMS.iter_no!=0):
    #var_info = np.load('./exps/oed/tile{}/digit{}_var{}_N{}/var_info.npy'.format(PARAMS.tile_size, PARAMS.digit, noise_var, N))
    var_info  = np.load(save_dir+'/../var_info.npy')    
    if (PARAMS.iter_no==1):
        tile_tl_row = int(var_info[PARAMS.iter_no-1,0])
        tile_tl_col = int(var_info[PARAMS.iter_no-1,1])
    else:
        rows = var_info[:PARAMS.iter_no,0].astype(np.int32)
        cols = var_info[:PARAMS.iter_no,1].astype(np.int32)


# tflow graph for posterior computation
with tf.Graph().as_default() as g:
    def joint_log_prob(z):         
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = gen_out - tf.constant(y_hat4d)
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), scale_diag=np.ones(z_dim, dtype=np.float32))
        dim_window = PARAMS.iter_no*(tile_size**2)

        if(PARAMS.iter_no==0):
            return prior.log_prob(z)
        elif(PARAMS.iter_no==1):            
            diff_img_visible = tf.reshape(tf.slice(diff_img, [0,tile_tl_row, tile_tl_col, 0], [1,tile_size,tile_size,1]), [int(dim_window/PARAMS.iter_no)])
            like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_window, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_window, dtype=np.float32))
            return (prior.log_prob(z) + like.log_prob(diff_img_visible))
        else:
            ls = []
            for ii in range(PARAMS.iter_no):
                ls.append(tf.reshape(tf.slice(diff_img, [0, rows[ii], cols[ii],0], [1,tile_size,tile_size,1]), [int(dim_window/PARAMS.iter_no)]))
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
    print('HMC acceptance ratio = {}'.format(sess.run(p_accept)))
