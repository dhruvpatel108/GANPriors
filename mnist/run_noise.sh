for digit in {0..9}    # digits 
do
    for noise in 0.1 0.5 1.0 2.0 10.0      # noise level
    do
        print digit=$digit and noise=$noise
        python mcmc_sampler.py --digit $digit --noise_var $noise
        python mcmc_stats.py --digit $digit --noise_var $noise
    done
done
