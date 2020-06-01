"""
code for training of WGAN-GP model

reused from https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/tree/v1
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from config import argparser

import utils
import traceback
import numpy as np
import tensorflow as tf
import data_mnist as data
import models_mnist as models

PARAMS = argparser()

""" param """
epoch = PARAMS.epoch
N = epoch
batch_size = PARAMS.train_batch_size
lr = PARAMS.lr
z_dim = PARAMS.z_dim
n_critic = PARAMS.n_critic
gpu_id = PARAMS.gpu_id
save_dir = PARAMS.prefix+'_N{}'.format(N)
''' data '''
utils.mkdir('./data/mnist/')
imgs, _, _ = data.mnist_load('./data/mnist')
imgs.shape = imgs.shape + (1,)
data_pool = utils.MemoryData({'img': imgs}, batch_size)


""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = models.generator
    discriminator = models.discriminator_wgan_gp

    ''' graph '''
    # inputs
    real = tf.placeholder(tf.float32, shape=[None, PARAMS.img_h, PARAMS.img_w, PARAMS.img_c])
    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    # generate
    fake = generator(z, reuse=False)

    # dicriminate
    r_logit = discriminator(real, reuse=False)
    f_logit = discriminator(fake)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    d_loss = -wd + gp * 10.0
    g_loss = -tf.reduce_mean(f_logit)

    # otpims
    d_var = utils.trainable_variables('discriminator')
    g_var = utils.trainable_variables('generator')
    d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})

    # sample
    f_sample = generator(z, training=False)


""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=5)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/'+save_dir, sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/'+save_dir
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    z_ipt_sample = np.random.normal(size=[batch_size, z_dim])

    batch_epoch = len(data_pool) // (batch_size * n_critic)
    max_it = epoch * batch_epoch
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // batch_epoch
        it_epoch = it % batch_epoch + 1

        # train D
        for i in range(n_critic):
            # batch data
            real_ipt = data_pool.batch('img')
            z_ipt = np.random.normal(size=[batch_size, z_dim])
            #_ = sess.run(assign_op, feed_dict={z: z_ipt})
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z:z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        z_ipt = np.random.normal(size=[batch_size, z_dim])
        #_ = sess.run(assign_op, feed_dict={z: z_ipt})
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z:z_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % PARAMS.log_freq == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % PARAMS.save_freq == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % PARAMS.sample_freq == 0:
            #_ = sess.run(assign_op, feed_dict={z: z_ipt_sample})
            f_sample_opt = sess.run(f_sample, feed_dict={z:z_ipt_sample})

            sample_dir = './sample_images_while_training/'+save_dir
            utils.mkdir(sample_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 8, 8), '%s/Epoch_(%d)_(%dof%d).jpg' % (sample_dir, epoch, it_epoch, batch_epoch))

except Exception:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()
