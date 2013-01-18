#! /bin/bash
# This is a simple script to cleanup the intermediate files in 
# spearmint experiment directories
[[ -n "$1" ]] || { echo "Usage: cleanup.sh <experiment_dir>"; exit 0 ; }
if [ -d $1 ]
then
    cd $1
    rm trace.csv
    rm output/*
    rm jobs/*
    rm expt-grid.pkl
    rm expt-grid.pkl.lock
    rm GP*Chooser*.pkl
    rm GPEIOptChooser*hyperparameters.txt
    rm best_job_and_result.txt
    rm results.dat
else
    echo "$1 is not a valid directory"
fi