for ii in {1..9}
do
    echo Angle = $ii
    python mcmc_sampler.py --digit 2 --noise_var 0.3 --angle $ii
    python mcmc_stats.py --digit 2 --noise_var 0.3 --angle $ii
done
