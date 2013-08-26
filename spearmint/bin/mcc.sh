#!/bin/bash

#module load  ....

MY_WRAPPER_SCRIPT=matlab_wrapper.m
export PATH=/sbin:$PATH
export DISPLAY=

# Here, copy all mfiles to root, and then create option string
#  to include all, of the format:
#
#  -a file1.m -a file2.m ...
#
ALL_MFILES=$(find . -name \*.m )


#
# Matlab compiler command
#
mcc -mv ${MY_WRAPPER_SCRIPT} ${ADD_FILE_CMD_OPTS}

#
# Grab MCR
#
tar cxvf mcr.tgz $MATLAB (stuff LD_LIBRARY_LINES in run_*.sh)

