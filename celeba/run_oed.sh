for ii in $(eval echo {0..$3})   # OED iteration
do
    echo img no = $1  iter_no = $ii
    python oed_sampler.py --img_no $1 --noise_var $2 --iter_no $ii
    python oed_stats.py --img_no $1 --noise_var $2 --iter_no $ii
done

