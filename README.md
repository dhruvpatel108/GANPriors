# GAN-based Priors for Quantifying Uncertainty
This is an official TensorFlow implementation of [GAN-based Priors for Quantifying Uncertainty](https://arxiv.org/abs/2003.12597)
work. Here, we provide code for our models, checkpoints and [dataset](https://github.com/dhruvpatel108/GANPriors/blob/master/README.md#downloading-and-preparing-dataset) used to produce results of the paper.

Bayesian inference is a powerful method used extensively to quantify the uncertainty in an inferred field given the measurement of a related field
when the two are linked by a mathematical model. Despite its many applications, Bayesian inference faces two major 
challenges: 
1. sampling from high dimensional posterior distributions and 
2. representing complex prior distributions that are difficult to characterize mathematically. 

In this work we demonstrate how the approximate distribution learned by a generative adversarial network (GAN) may be used as a
prior in a Bayesian update to address both these challenges.

We demonstrate the efficacy of this approach by inferring and 
quantifying uncertainty in inference problems arising in computer vision and physics-based applications. 
In both instances we highlight the role of computing uncertainty in providing a measure of confidence in the solution,
and in designing successive measurements to improve this confidence. 

Following animation shows how the proposed method can be used in active learning/design of experiments setting in deciding
optimal sensor placement location:

CelebA             |  MNIST
:-------------------------:|:-------------------------:
![](https://github.com/dhruvpatel108/GANPriors/blob/master/images/celeba_oed.gif)  |  ![](https://github.com/dhruvpatel108/GANPriors/blob/master/images/mnist_oed.gif)

Following figures demonstrate the effectiveness of the proposed method (in inferring field and quantifying uncertainty) for
various inverse problems arising in computer vision and physics-based applications: 
#### Image inpainting
<p align="center">
  <img width="460" height="300" src="https://github.com/dhruvpatel108/GANPriors/blob/master/images/mnist_inpaint.png">
</p>

#### Image recovery/denoising
<p align="center">
  <img width="460" height="300" src="https://github.com/dhruvpatel108/GANPriors/blob/master/images/celeba_recovery.png">
</p>

#### Initial condition inversion (heat conduction)
<p align="center">
  <img width="360" height="200" src="https://github.com/dhruvpatel108/GANPriors/blob/master/images/ic_inversion.png">
</p>

## Requirements
* Python 3.7
* [Numpy](https://numpy.org/)
* [TensorFlow 1.14.0](https://github.com/tensorflow/tensorflow/releases)
* [TensorFlow Probability 0.7.0](https://github.com/tensorflow/probability/releases)
* [Scipy](https://www.scipy.org/install.html)

## Usage
### Downloading and preparing dataset
After cloning the repo, first change directory to the dataset of interest and follow the steps below.
* *CelebA:*
  ```
  $ python data_celeba.py --download_path data
  ```
  The above command will download the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset and split it into training and test set and store it in the directory specified by `--download_path` argument.
* *MNIST:*

  Run `trainer.py` script as explained [below](https://github.com/dhruvpatel108/GANPriors/blob/master/README.md#learning-the-prior-distribution-training-of-gan). It will automatically download MNIST dataset and split it into training and test set before proceeding to training.

### Training 
The proposed algorithm works in two steps: In first step, we train a GAN on a particular dataset to learn the prior distribution and in second step depending upon task at hand, we perform different inference steps.
 
#### Learning the prior distribution (training of GAN)
```
$ python trainer.py --epoch 100 --z_dim 100 --train_batch_size 64 --lr 0.001
```
* Selected arguments (see `config.py` for more details):
  * Hyperparameters
    * --n_critic: no. of discriminator updates for every generator update
    * --train_batch_size: batch size during training (note that this is different than the batch size during posterior inference)
    * --epoch: the max training epochs
  * Logging
    * --log_freq: the frequency of printing out log info (default 1)
    * --save_freq: the frequency of saving a checkpoint (default 1000)
    * --sample_freq: the frequency of performing testing inference to save generated fake images during training (default 100)
  * --gpu_id: id of the gpu to be used for training
  * --prefix: a nickname for the training

#### Posterior inference
We propose different methods for probing posterior distribution in efficient way for different inference tasks.
 1. *Image recovery/denoising*:

    Here the goal is to infer the true field and associated uncertainty from a noisy measurement.
    ```
    $ python mcmc_sampler.py --digit 5 --noise_var 0.1
    $ python mcmc_stats.py --digit 5 --noise_var 0.1
    ```
    * Selected arguments (see `config.py` for more details):
      * --digit: an MNIST digit of interest (choose from `[0,1,2,..,9]`)
      * --noise_var: variance of additive Gaussian noise in measurement
      * --n_mcmc: no. of MCMC samples
      * --batch_size: batch size for MCMC inference (should always be one)
      
    For CelebA, use `--img_no` argument instead of `--digit` (see `config.py` for details).
 
    
 2. *Image inpainting*:

    Here the goal is to infer the true field and associated uncertainty from a noisy and occluded measurement.
    ```
    $ python inpaint_sampler.py --digit 5 --noise_var 0.1
    $ python inpaint_stats.py --digit 5 --noise_var 0.1
    ```
    * Selected arguments (see `config.py` for more details):
      * --start_row: row index of top left corner of mask
      * --end_row: row index of bottom right corner of mask
      * --start_col: column index of top left corner of mask
      * --end_col: column index of bottom right corner of mask
     
      
    For CelebA, use `--img_no` argument instead of `--digit` (see `config.py` for details).

### Cite the paper
If you find this useful, please cite
```
@article{Patel2020b,
archivePrefix = {arXiv},
arxivId = {2003.12597},
author = {Patel, Dhruv V. and Oberai, Assad A.},
doi = {10.13140/RG.2.2.28806.32322},
title = {{GAN-based Priors for Quantifying Uncertainty}},
url = {http://arxiv.org/abs/2003.12597 http://dx.doi.org/10.13140/RG.2.2.28806.32322},
year = {2020}
}
