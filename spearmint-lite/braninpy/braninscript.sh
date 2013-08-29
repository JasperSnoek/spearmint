#! /bin/bash
# Jasper Snoek
# This is a simple script to help demonstrate the functionality
# of spearmint-lite.  This runs the braninpy experiment automatically
# within the framework of spearmint lite.  It will iteratively get 
# spearmint-lite to propose 3 new experiments, fill in the results, and repeat.
for i in {1..50}
do
    cd ..
    for i in {1..2}
    do
	python spearmint-lite.py --method=GPEIOptChooser --grid-size=20000 --method-args=mcmc_iters=10,noiseless=1 braninpy
    done

    cd braninpy
    python braninrunner.py
done