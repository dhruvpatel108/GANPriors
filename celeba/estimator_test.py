import time
import glob
import tensorflow as tf
import numpy as np
import utils
import tensorflow.contrib.slim as slim
from models_64x64 import generator as gen
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import matplotlib.patches as patches
import matplotlib

tf.reset_default_graph()
N = 64000
batch_size = 64
z_dim = 100
noise_lvl = 0.8
like_stddev = noise_lvl
dim_prior = z_dim
prior_stddev = 1.0
seed = 1008
img_no = 202560 
st_time = time.time()

n_iter = int(N/batch_size)
numerator = np.zeros((batch_size, n_iter))

loss_value = np.zeros((batch_size, n_iter))

#x_rec = np.zeros((batch_size, 64, 64, 3, n_iter))
z_store = np.zeros((batch_size, z_dim, n_iter))

''' data '''
def preprocess_fn(img):
    crop_size = 108
    re_size = 64
    img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
    img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 - 1
    return img

img_paths = glob.glob('./data/test/{}.jpg'.format(img_no))
data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)

np.random.seed(seed)
dim_like = 64*64*3
noise_mat3d = np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*like_stddev, size=1)
np.save('noise_mat3d_seed{}'.format(seed), noise_mat3d)
noisy_mat3d = data_pool.batch()[0,:,:,:] + np.reshape(noise_mat3d, [64,64,3])
noisy_mat4d = np.tile(noisy_mat3d, (batch_size, 1, 1, 1)).astype(np.float32)



with tf.Graph().as_default() as g:
    z1 = tf.placeholder(tf.float32, shape=[batch_size, z_dim])    
    dummy = tf.Variable(name='dummy', trainable=False, initial_value=z1)   
    assign_op1 = dummy.assign(z1)
    
    gen_out = gen(z1, reuse=tf.AUTO_REUSE, training=False)
    diff_img = gen_out - tf.constant(noisy_mat4d)
    
    #init_op = tf.initializers.global_variables()
    
    variables = slim.get_variables_to_restore()    
    variables_to_restore = [v for v in variables if v.name.split(':')[0]!='dummy'] 
    
    saver = tf.train.Saver(variables_to_restore)

z_sample = np.random.normal(size=[batch_size, z_dim])
with tf.Session(graph=g) as sess:
    #sess.run(init_op, feed_dict={z1:z_sample})
    _ = sess.run(assign_op1, feed_dict={z1:z_sample})  
    
    saver.restore(sess, PARAMS.model_path)
    print('Model restored!')
    
    
    """
    save_dir = './sample_test/mnist_wgan_gp{}'.format(N)
    #utils.mkdir(save_dir + '/')
    
    for it in range(5):
        z_sample = np.random.normal(size=[batch_size, z_dim]).astype(np.float32)
        _ = sess.run(assign_op1, feed_dict={z1:z_sample})  
        gen_out_, diff_img_ = sess.run([gen_out, diff_img])
        #utils.imwrite(utils.immerge(noisy_mat4d, 10, 10), '{}/214_noisy.jpg'.format(save_dir))# % (save_dir, epoch, it_epoch, batch_epoch))
        utils.imwrite(utils.immerge(gen_out_, 8, 8), '{}/121_it{}.jpg'.format(save_dir, it))
        utils.imwrite(utils.immerge(diff_img_, 8, 8), '{}/121diff_it{}.jpg'.format(save_dir, it))
    """   
        
    
    for n in range(n_iter):
        z_sample = np.random.normal(size=[batch_size, z_dim])
        z_store[:,:,n] = z_sample
        gen_out_, diff_img_ = sess.run([gen_out, diff_img], feed_dict={z1:z_sample})
        #mask = np.ones((batch_size,64,64,3), dtype=bool)
        #mask[:,20:40, 20:40,:] = False
        #masked_out = diff_img_[mask]
        if (n%100)==0:
            print('iterations={}'.format(n))
        for k in range(batch_size):
            sample_norm2 = np.linalg.norm(diff_img_[k,:,:,:])**2
            #sample_norm2 = np.linalg.norm(masked_out[k*684:(k+1)*684])**2
            #print('img{} norm2={}'.format(k+1, sample_norm2))
            numerator[k,n] = np.exp(-(sample_norm2/(2*like_stddev)))
            loss_value[k,n] = ((0.5*sample_norm2)/(like_stddev)) + (0.5*np.linalg.norm(z_sample[k,:])**2)   
            #print('img{} norm2={}'.format(k+1, numerator[k, n]))
        #utils.imwrite(utils.immerge(gen_out_, 80, 80), '{}/001_it{}.jpg'.format(save_dir, n))
        #utils.imwrite(utils.immerge(diff_img_, 80, 80), '{}/001diff_it{}.jpg'.format(save_dir, n))
        #print('max = {}'.np.argmax())        
        
    print(np.max(numerator))
    print(np.mean(numerator))
    norm_prob = numerator/np.sum(numerator)
    print(np.max(norm_prob))
    print(np.mean(norm_prob))
    k_i, n_i = np.unravel_index(loss_value.argmin(), loss_value.shape)
    k_i_post, n_i_post = np.unravel_index(norm_prob.argmax(), norm_prob.shape)
    
    save_dir = './sample_test/monte_carlo/test/celeba_wgan_gp{}_imgno{}_noisevar{}_seed{}'.format(N,img_no,like_stddev,seed)    
    utils.mkdir(save_dir+'/')
    #n_img_per_row = int(np.floor(np.sqrt(batch_size)))
    #utils.imwrite(utils.immerge(x_map, 8, 8), '{}/xmap.jpg'.format(save_dir))
    
    
    x_map4d = sess.run(gen_out, feed_dict={z1:z_store[:,:,n_i_post]})
    x_map = x_map4d[k_i_post,:,:,:]   
    x_map = np.tile(x_map, (batch_size, 1, 1, 1)).astype(np.float32) 
    
    
    #fig, axs = plt.subplots()
    #cb = axs.imshow(x_map[0,:,:,0], cmap='viridis')
    #axs.title.set_text('x_map_MC -  N={}'.format(N))
    #axs.add_patch(patches.Rectangle(10,10,10,10, linewidth=1, edgecolor='r', facecolor='none'))
    #plt.colorbar(cb, ax=axs)
    fig = plt.figure()
    axs = fig.add_subplot(111)
    cb = plt.imshow(x_map[0,:,:,:], cmap='viridis')
    plt.colorbar(cb, ax=axs)
    plt.title('x_map | N={}'.format(N))
    plt.savefig('./{}/xmap_N{}_LikeVar{}.png'.format(save_dir, N, like_stddev))
    
    #assert (k_i==k_i_post and n_i==n_i_post), "check max posterior and min loss!"
    
    
    x_mean = np.zeros((64,64,3))
    x2_mean = np.zeros((64,64,3))
    for n in range(n_iter):
        x_out = sess.run(gen_out, feed_dict={z1:z_store[:,:,n]})
        
        x_mean +=  np.dot(np.transpose(x_out, axes=(1,2,3,0)), norm_prob[:,n])
        x2_mean += np.dot(np.transpose(x_out**2, axes=(1,2,3,0)), norm_prob[:,n])
    
    var = x2_mean - (x_mean**2)
    
    prob_reshape = np.reshape(norm_prob, [batch_size*n_iter, 1])
    descending = np.sort(prob_reshape, axis=0)[::-1]
    cummulative = np.cumsum(descending)
    effective_samps = int(sum((cummulative<0.99).astype(int)))+1    
    print('sum of normailized prob. = {} and shape of it is {}'.format(np.sum(norm_prob), np.shape(norm_prob)))
    print('no. of non-zero norm_probs = {}'.format(np.size(np.nonzero(norm_prob))))
    print('max. norm-prob = {}'.format(np.max(norm_prob)))
    print('no. of effective samples(cumm. prob. <0.99) = {}'.format(effective_samps))
    #print('normalized probability is descending order={}'.format(descending))
    #np.savez('./sample_test/monte_carlo/mnist_wgan_gp{}/data'.format(N), norm_prob=norm_prob, x_map=x_map, x_mean=x_mean, var=var)
    #fig2, axs2 = plt.subplots()
    #cb2 = axs2.imshow(x_mean[:,:,0], cmap='viridis')
    #axs2.title.set_text('x_mean_MC - N={}'.format(N))
    #axs2.add_patch(patches.Rectangle(10,10,10,10, linewidth=1,edgecolor='r',facecolor='none'))
    #plt.colorbar(cb2, ax=axs)
    fig = plt.figure()
    axs = fig.add_subplot(111)
    cb = plt.imshow(x_mean, cmap='viridis')
    plt.colorbar(cb, ax=axs)
    plt.title('x_mean -  N={} | n_eff={}'.format(N, effective_samps))
    plt.savefig('./{}/xmean_N{}_effsamples{}'.format(save_dir, N, effective_samps))
    #plt.show()  
    
    #fig, axs = plt.subplots()
    #cb = axs.imshow(var[:,:,0], cmap='viridis')
    #axs.title.set_text('var_MC - LikeVar={}, N={}'.format(like_stddev, N))
    #axs.add_patch(patches.Rectangle(10,10,10,10, linewidth=1, edgecolor='r', facecolor='none'))
    #plt.colorbar(cb, ax=axs)
    fig = plt.figure()
    axs = fig.add_subplot(111)
    cb = plt.imshow(var, cmap='viridis')
    plt.colorbar(cb, ax=axs)
    plt.title('var - N={} | n_eff={}'.format(like_stddev, N, effective_samps))
    plt.savefig('./{}/var_N{}_effsamples{}'.format(save_dir, N, effective_samps))
    
    
    plt.figure()
    plt.imshow(data_pool.batch()[0,:,:,:])
    plt.colorbar()
    plt.title('x')
    plt.savefig('./{}/x'.format(save_dir))

    #fig, axs = plt.subplots()
    #cb = axs.imshow(noisy_mat4d[0,:,:,0])
    #axs.title.set_text('y')    
    #axs.add_patch(patches.Rectangle(10,10,10,10, linewidth=1,edgecolor='r',facecolor='white'))
    #axs.grid(True)
    #plt.savefig('./{}/y'.format(save_dir))
    #plt.colorbar(cb, ax=axs)
    plt.figure()
    plt.imshow(noisy_mat4d[0,:,:,:], cmap='viridis')
    plt.colorbar()
    plt.savefig('./{}/y_noisevar{}.png'.format(save_dir, like_stddev))
    #plt.show()
    end_time = time.time()
    print('total run time = {}'.format(end_time-st_time))
    
    #np.save(save_dir+'norm_prob.npy', norm_prob)    
    #np.save(save_dir+'x_map.npy', x_map) 
    #np.save(save_dir+'x_mean.npy', x_mean)
    #np.save(save_dir+'var.npy', var) 
