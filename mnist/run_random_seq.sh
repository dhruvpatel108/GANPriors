for i in {9}  # iteration_no 
do
    echo iter_no = $i
    python random_seq_sampler.py --digit 6 --noise_var 1.0 --iter_no $i
    python random_seq_stats.py --digit 6 --noise_var 1.0 --iter_no $i
done

