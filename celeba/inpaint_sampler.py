import os
import utils
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
import tensorflow_probability as tfp
from config import argparser
from data_process import noisy_meas


tfd= tfp.distributions
tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)


N = PARAMS.n_inpaint
burn = int(PARAMS.burn_inpaint*N)
z_dim = PARAMS.z_dim
batch_size = PARAMS.batch_size # should always be one
noise_var = PARAMS.noise_var
dim_prior = z_dim
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
img_no = PARAMS.img_no

save_dir = './exps/inpaint/img{}_var{}_N{}_strow{}_endrow{}_stcol{}_endcol{}'.format(PARAMS.img_no, noise_var, N, 
                                                                            PARAMS.start_row, PARAMS.end_row, 
                                                                            PARAMS.start_col, PARAMS.end_col)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  

x_true, noisy_mat4d = noisy_meas(batch_size)

mask = np.ones((batch_size, PARAMS.img_h, PARAMS.img_w, PARAMS.img_c))
mask[:, PARAMS.start_row:PARAMS.end_row, PARAMS.start_col:PARAMS.end_col, :] = 0.
dim_inpaint = int(np.sum(mask))

with tf.Graph().as_default() as g:
    def joint_log_prob(z):              
        
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = gen_out - tf.constant(noisy_mat4d)
        visible_img = tf.boolean_mask(diff_img, mask)
        
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), scale_diag=np.ones(z_dim, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_inpaint, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_inpaint, dtype=np.float32))
        
        return (prior.log_prob(z) + like.log_prob(visible_img))
                                          

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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    samples_ = sess.run(samples)
    np.save(save_dir+'/samples.npy', samples_)
    print('acceptance ratio = {}'.format(sess.run(p_accept)))
