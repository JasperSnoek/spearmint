##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled "Practical Bayesian Optimization of
# Machine Learning Algorithms" by Snoek, Larochelle and Adams Advances
# in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
import optparse
import tempfile
import datetime
import subprocess
import time
import imp
import os
import re
import collections
import importlib

from ExperimentGrid  import *
try: import simplejson as json
except ImportError: import json

#
# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.
#
# The spearmint.py script can run in two modes, which reflect experiments
# vs jobs.  When run with the --wrapper argument, it will try to run a
# single job.  This is not meant to be run by hand, but is intended to be
# run by a job queueing system.  Without this argument, it runs in its main
# controller mode, which determines the jobs that should be executed and
# submits them to the queueing system.
#

def main():
    parser = optparse.OptionParser(usage="usage: %prog [options] directory")

    parser.add_option("--n", dest="num_jobs",
                      help="Number of concurrent jobs to create.",
                      type="int", default=1)
    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=1000)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments.",
                      type="string", default="GPEIOptChooser")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=1000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("--config", dest="config_file",
                      help="Configuration file name.",
                      type="string", default="config.json")
    parser.add_option("--results", dest="results_file",
                      help="Results file name.",
                      type="string", default="results.dat")

    (options, args) = parser.parse_args()

    # Otherwise run in controller mode.
    main_controller(options, args)

##############################################################################
##############################################################################
def main_controller(options, args):

    expt_dir  = os.path.realpath(args[0])
    work_dir  = os.path.realpath('.')
    expt_name = os.path.basename(expt_dir)

    if not os.path.exists(expt_dir):
        sys.stderr.write("Cannot find experiment directory '%s'.  Aborting.\n" % (expt_dir))
        sys.exit(-1)

    # Load up the chooser module.
    module  = importlib.import_module('chooser.' + options.chooser_module, package='spearmint')
    chooser = module.init(expt_dir, options.chooser_args)

    # Create the experimental grid
    expt_file = os.path.join(expt_dir, options.config_file)
    variables = json.load(open(expt_file), object_pairs_hook=collections.OrderedDict)

    #@gdahl - added the following three lines and commented out the line above
    vkeys = [k for k in variables]
    #vkeys.sort()
    gmap = GridMap([variables[k] for k in vkeys], options.grid_size)

    # Read in parameters and values observed so far
    for i in xrange(0,options.num_jobs):

        res_file = os.path.join(expt_dir, options.results_file)
        if not os.path.exists(res_file):
            thefile = open(res_file, 'w')
            thefile.write("")
            thefile.close()

        values = np.array([])
        complete = np.array([])
        pending = np.array([])
        durations = np.array([])
        index = 0

        infile = open(res_file, 'r')
        for line in infile.readlines():
            # Each line in this file represents an experiment
            # It is whitespace separated and of the form either
            # <Value> <time taken> <space separated list of parameters>
            # incating a completed experiment or
            # P P <space separated list of parameters>
            # indicating a pending experiment
            expt = line.split()
            if (len(expt) < 3):
                continue

            val = expt.pop(0)
            dur = expt.pop(0)
            variables = gmap.to_unit(expt)
            if val == 'P':
                if pending.shape[0] > 0:
                    pending = np.vstack((pending, variables))
                else:
                    pending = np.matrix(variables)
            else:
                if complete.shape[0] > 0:
                    values = np.vstack((values, float(val)))
                    complete = np.vstack((complete, variables))
                    durations = np.vstack((durations, float(dur)))
                else:
                    values = float(val)
                    complete = np.matrix(variables)
                    durations = float(dur)

        infile.close()
        # Some stats
        sys.stderr.write("#Complete: %d #Pending: %d\n" %
                         (complete.shape[0], pending.shape[0]))

        # Let's print out the best value so far
        if type(values) is not float and len(values) > 0:
            best_val = np.min(values)
            best_job = np.argmin(values)
            sys.stderr.write("Current best: %f (job %d)\n" % (best_val, best_job))

        # Now lets get the next job to run
        # First throw out a set of candidates on the unit hypercube
        # Increment by the number of observed so we don't take the
        # same values twice
        off = pending.shape[0] + complete.shape[0]
        candidates = gmap.hypercube_grid(options.grid_size,
                                         options.grid_seed+off)

        # Ask the chooser to actually pick one.
        # First mash the data into a format that matches that of the other
        # spearmint drivers to pass to the chooser modules.
        grid = candidates
        if (complete.shape[0] > 0):
            grid = np.vstack((complete, candidates))
        if (pending.shape[0] > 0):
            grid = np.vstack((grid, pending))
        grid = np.asarray(grid)
        grid_idx = np.hstack((np.zeros(complete.shape[0]),
                              np.ones(candidates.shape[0]),
                              1.+np.ones(pending.shape[0])))
        job_id = chooser.next(grid, np.squeeze(values), durations,
                              np.nonzero(grid_idx == 1)[0],
                              np.nonzero(grid_idx == 2)[0],
                              np.nonzero(grid_idx == 0)[0])

        # If the job_id is a tuple, then the chooser picked a new job not from
        # the candidate list
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
        else:
            candidate = grid[job_id,:]

        sys.stderr.write("Selected job %d from the grid.\n" % (job_id))
        if pending.shape[0] > 0:
            pending = np.vstack((pending, candidate))
        else:
            pending = np.matrix(candidate)

        params = gmap.unit_to_list(candidate)

        # Now lets write this candidate to the file as pending
        output = ""
        for p in params:
            output = output + str(p) + " "

        output = "P P " + output + "\n"
        outfile = open(res_file,"a")
        outfile.write(output)
        outfile.close()

# And that's it
if __name__ == '__main__':
    main()
