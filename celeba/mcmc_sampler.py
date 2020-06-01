import os
import glob
import utils
import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from config import argparser
from data_process import noisy_meas

tfd= tfp.distributions
tf.reset_default_graph()
PARAMS = argparser()
np.random.seed(PARAMS.seed_no)

N = PARAMS.n_mcmc
burn = int(PARAMS.burn_mcmc*N)
z_dim = PARAMS.z_dim
batch_size = PARAMS.batch_size # should always be one
noise_var = PARAMS.noise_var
dim_prior = z_dim
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
img_no = PARAMS.img_no


"""
''' data '''
def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/test/{}.jpg'.format(img_no))
noisefile_path = './data/noise3d_var{}_seed{}.npy'.format(noise_var, seed_no)
if not os.path.exists(noisefile_path):
    print('**** noise file does not exist... creating new one ****')
    noise_mat3d = np.reshape(np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1), (64,64,3))
    np.save(noisefile_path, noise_mat3d)
noise_mat3d = np.load(noisefile_path)
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)

noisy_meas3d = data_pool.batch()[0,:,:,:] + noise_mat3d
noisy_mat4d = np.tile(noisy_meas3d, (batch_size, 1, 1, 1)).astype(np.float32)
"""

x_true3d, noisy_mat4d = noisy_meas(batch_size)
save_dir = './exps/mcmc/img{}_var{}_N{}'.format(img_no, noise_var, N)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


with tf.Graph().as_default() as g:
    def joint_log_prob(z):              
        
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = tf.reshape(gen_out - tf.constant(noisy_mat4d), [dim_like])
                
        
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), scale_diag=np.ones(z_dim, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(dim_like, dtype=np.float32), scale_diag=np.sqrt(noise_var)*np.ones(dim_like, dtype=np.float32))
        
        return (prior.log_prob(z) + like.log_prob(diff_img))
                                          

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
    utils.mkdir(save_dir+'/')
    np.save(save_dir+'/samples.npy', samples_)
    print('acceptance ratio = {}'.format(sess.run(p_accept)))
