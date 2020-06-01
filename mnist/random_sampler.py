import utils
import argparse
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from models_mnist import generator as gen
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

tfd= tfp.distributions

tf.reset_default_graph()
N = 64000
burn = int(0.5*N)
z_dim = 100
batch_size = 1 # should always be one
noise_lvl = 1.0
dim_prior = z_dim
dim_like = 28*28*1
random_sampling = True

prior_stddev = 1.0
parser = argparse.ArgumentParser()
parser.add_argument('--digit', type=int, default=1)
parser.add_argument('--noise_var', type=float, default=noise_lvl)
parser.add_argument('--random_seed', type=int, default=1008)
parser.add_argument('--n_meas', type=int, default=20)
args = parser.parse_args()

print('------------- digit_in = {} ---------------'.format(args.digit))
print('------------- noise_var = {} ---------------'.format(args.noise_var))
print('------------- random_seed = {} ---------------'.format(args.random_seed))
print('------------- n_meas = {}----------------/'.format(args.n_meas))

''' data '''
digit_idx=[3,2,1,18,4,8,11,0,61,7]

test_data = np.load('test_set/test_x.npy')
test_sample = np.reshape(test_data[digit_idx[args.digit]], [28,28,1])
np.random.seed(args.random_seed)
noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*args.noise_var, size=1)
noisy_mat3d = test_sample + np.reshape(noise_mat3d, [28,28,1])
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)
np.save('./mcmc/random_sampling/digit{}_var{}_N{}.npy'.format(args.digit,args.noise_var, N), noisy_mat4d)
indices = np.random.choice(dim_like, size=(args.n_meas,1), replace=False)

with tf.Graph().as_default() as g:
    def joint_log_prob(z):              
        gen_out = gen(z, reuse=tf.AUTO_REUSE, training=False)
        diff_img = tf.reshape(gen_out - tf.constant(noisy_mat4d), [dim_like])
        
        if random_sampling==True:
            diff_img = tf.gather_nd(diff_img, indices)
        
        prior = tfd.MultivariateNormalDiag(loc=np.zeros(z_dim, dtype=np.float32), 
                                           scale_diag=np.ones(z_dim, dtype=np.float32))
        like = tfd.MultivariateNormalDiag(loc=np.zeros(args.n_meas, dtype=np.float32), 
                                          scale_diag=np.sqrt(args.noise_var)*np.ones(args.n_meas, dtype=np.float32))
        
        return (prior.log_prob(z) + like.log_prob(diff_img))
                                          

    def unnormalized_posterior(z):
        return joint_log_prob(z) 
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                    tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=unnormalized_posterior, step_size=np.float32(1.8), num_leapfrog_steps=3),
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
    #variables_to_restore = [v for v in variables if v.name.split(':')[0]!='dummy'] 
    saver = tf.train.Saver(variables_to_restore)
    

with tf.Session(graph=g) as sess:
    
    saver.restore(sess, model_path)
    
    samples_ = sess.run(samples)
    save_dir = './mcmc/random_sampling/digit{}_var{}_nMeas{}_N{}'.format(args.digit, args.noise_var, args.n_meas, N)
    utils.mkdir(save_dir+'/')
    np.save(save_dir+'/samples.npy', samples_)
    np.save(save_dir+'/noisy_mat4d.npy', noisy_mat4d)
    
    mm = np.ones(784)
    mm[indices] = 0.
    mm = np.reshape(mm, (28,28))
    masked_y = np.ma.masked_array(noisy_mat4d, mask=mm)

    np.save(save_dir+'/masked_y.npy', masked_y)
    np.save(save_dir+'/indices.npy', indices)
    #np.save(save_dir+'/)
    #np.save(save_dir+'/noisy_mat4d.npy', noisy_mat4d)
    #print('acceptance ratio = {}'.format(sess.run(p_accept)))
    """
    plt.figure(figsize=(15, 6))
    for ii in range(25):
        plt.subplot(5,5,ii+1)
        plt.hist(samples_[:, 0, ii], 50, density=True);
        plt.xlabel(r'$x_{}$'.format(ii))
    plt.tight_layout()
    plt.savefig('./{}/hist_total_samples'.format(save_dir))
    plt.figure(figsize=(15, 6))
    for ii in range(25):
        plt.subplot(5,5,ii+1)
        plt.plot(samples_[:, 0, ii]);
        plt.ylabel(r'$x_{}$'.format(ii))
    plt.tight_layout()
    plt.savefig('./{}/total_samples'.format(save_dir))

    eff_samples = np.squeeze(samples_[burn:,:,:])
    
    plt.figure(figsize=(15, 6))
    for ii in range(25):
        plt.subplot(5,5,ii+1)
        plt.hist(eff_samples[:, ii], 50, density=True);
        plt.xlabel(r'x_{} eff samples'.format(ii))
    plt.tight_layout()
    plt.savefig('./{}/hist_eff_samples'.format(save_dir))
    plt.figure(figsize=(15, 6))
    for ii in range(25):
        plt.subplot(5,5,ii+1)
        plt.plot(eff_samples[:, ii]);
        plt.ylabel(r'x_{} eff samples'.format(ii))
    plt.tight_layout()
    plt.savefig('./{}/eff_samples'.format(save_dir))
    plt.show()

    # ------------ Mean and Variance from MCMC samples --------------------
    
    x_mcmc = sess.run(gen_out1, feed_dict={zz:eff_samples})
    
    x2_mcmc = x_mcmc**2
    x_mean = np.mean(x_mcmc, axis=0)
    x2_mean = np.mean(x2_mcmc, axis=0)
    var = x2_mean - (x_mean**2)
    
    plt.figure()
    plt.imshow(test_sample[:,:,0], cmap='viridis')
    plt.colorbar()
    plt.title('x')
    plt.figure()
    plt.imshow(noisy_mat4d[0,:,:,0], cmap='viridis')
    plt.colorbar()
    plt.figure()
    plt.imshow(x_mean[:,:,0], cmap='viridis')
    plt.colorbar()
    plt.title('mean')
    plt.figure()
    plt.imshow(var[:,:,0], cmap='viridis')
    plt.colorbar()
    plt.title('variance')
    plt.show()
    """
