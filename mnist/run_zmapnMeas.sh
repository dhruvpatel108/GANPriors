for noise_var in 1  # noise_level
do
    for digit in 2  # digit
    do
        for nMeas in 49 98 147 196 245 294 343 392 441 490 539 588  # no. of pixels (should be in the multiple of (tile_size)**2)
        do
            for iter_no in {1..20}
            do
                echo iter_no = $iter_no, digit = $digit, nMeas = $nMeas, noise_var = $noise_var
                python z_map_nMeas.py --iter_no $iter_no --digit $digit --nMeas $nMeas --noise_var $noise_var
            done
        done
    done
done
