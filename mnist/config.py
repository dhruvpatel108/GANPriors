import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Hyper-parameters')

    # == model parameters ==
    # GAN training 
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size for GAN training')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--n_critic', type=int, default=3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--z_dim', type=int, default=100)   
    parser.add_argument('--seed_no', type=int, default=1008)
    parser.add_argument('--img_h', type=int, default=28)
    parser.add_argument('--img_w', type=int, default=28)
    parser.add_argument('--img_c', type=int, default=1)
    parser.add_argument('--prefix', type=str, default='mnist_wgan_gp', help='nickname for training')
    parser.add_argument('--model_path', type=str, default = './checkpoints/Epoch_(999)_(171of171).ckpt')    
    parser.add_argument('--save_freq', type=int, default=1000, help='frequency of saving the model')
    parser.add_argument('--sample_freq', type=int, default=100, help='frequency of saving fake images')
    parser.add_argument('--log_freq', type=int, default=1)

    # measurement parameters for inverse problem
    parser.add_argument('--noise_var', type=float)
    parser.add_argument('--digit', type=int)
    parser.add_argument('--digit_array', type=int, nargs='+', default = [10,2,1,32,4,15,11,0,128,12], help='index of different MNIST digits in test set in the order [0,1,..,9]')
    
    # parameters for mcmc 
    parser.add_argument('--n_mcmc', type=int, default=64000)    
    parser.add_argument('--burn_mcmc', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for mcmc sampling')
    
    # parameters for oed experiments
    parser.add_argument('--n_oed', type=int, default=64000)    
    parser.add_argument('--burn_oed', type=float, default=0.5)    
    parser.add_argument('--iter_no', type=int)
    parser.add_argument('--tile_size', type=int, default=7)
    parser.add_argument('--n_random_meas', type=int, default=49)
    
    # parameters for inpainting experiments
    parser.add_argument('--n_inpaint', type=int, default=64000)
    parser.add_argument('--burn_inpaint', type=float, default=0.5)
    parser.add_argument('--start_row', type=int)
    parser.add_argument('--end_row', type=int)
    parser.add_argument('--start_col', type=int)
    parser.add_argument('--end_col', type=int)
    
    config = parser.parse_args()
    return config
