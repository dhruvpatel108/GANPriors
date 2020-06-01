import os
import glob
import utils
import numpy as np
import tensorflow as tf
from config import argparser
import matplotlib.pyplot as plt

PARAMS = argparser()
dim_like = PARAMS.img_h*PARAMS.img_w*PARAMS.img_c
noise_var = PARAMS.noise_var
seed_no = PARAMS.seed_no

def noisy_meas(batch_size): 
    ''' data '''
    def preprocess_fn(img):
        crop_size = 108
        re_size = 64
        img = tf.image.crop_to_bounding_box(img, (218 - crop_size) // 2, (178 - crop_size) // 2, crop_size, crop_size)
        img = tf.to_float(tf.image.resize_images(img, [re_size, re_size], method=tf.image.ResizeMethod.BICUBIC)) / 127.5 -     1
        return img

    img_paths = glob.glob('./data/test/{}.jpg'.format(PARAMS.img_no))
    noisefile_path = './meas/noise3d_var{}_seed{}.npy'.format(noise_var, seed_no)


    if not os.path.exists(noisefile_path):
        print('**** noise file does not exist... creating new one ****')
        noise_mat3d = np.reshape(np.random.multivariate_normal(mean=np.zeros((dim_like)), cov=np.eye(dim_like, dim_like)*noise_var, size=1), (64,64,3))
        np.save(noisefile_path, noise_mat3d)
    noise_mat3d = np.load(noisefile_path)
    data_pool = utils.DiskImageData(img_paths, batch_size, shape=[218, 178, 3], preprocess_fn=preprocess_fn)
    noisy_meas3d = data_pool.batch()[0,:,:,:] + noise_mat3d
    noisy_mat4d = np.tile(noisy_meas3d, (batch_size, 1, 1, 1)).astype(np.float32)
    return data_pool.batch()[0,:,:,:], noisy_mat4d
