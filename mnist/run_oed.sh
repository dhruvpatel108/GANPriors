for ii in $(eval echo {0..$3})   # OED iteration
do
    echo digit = $1 iter_no = $ii
    #python oed_sampler.py --digit $1 --noise_var $2 --iter_no $ii
    #python oed_stats.py --digit $1 --noise_var $2 --iter_no $ii
done
