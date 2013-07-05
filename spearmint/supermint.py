##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#
# This code is written for research and educational purposes only to
# supplement the paper entitled
# "Practical Bayesian Optimization of Machine Learning Algorithms"
# by Snoek, Larochelle and Adams
# Advances in Neural Information Processing Systems, 2012
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import optparse
import tempfile
import datetime
import subprocess
import multiprocessing
import importlib
import time
import imp
import os
import re
import Locker

from ExperimentGrid  import *
from helpers         import *
from runner          import job_runner


# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.
#
# The spearmint.py script can run in two modes, which reflect experiments
# vs jobs.  When run with the --run-job argument, it will try to run a
# single job.  This is not meant to be run by hand, but is intended to be
# run by a job queueing system.  Without this argument, it runs in its main
# controller mode, which determines the jobs that should be executed and
# submits them to the queueing system.


def parse_args():
    parser = optparse.OptionParser(usage="usage: %prog [options] directory")

    parser.add_option("--max-concurrent", dest="max_concurrent",
                      help="Maximum number of concurrent jobs.",
                      type="int", default=1)
    parser.add_option("--max-finished-jobs", dest="max_finished_jobs",
                      type="int", default=10000)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments.",
                      type="string", default="GPEIOptChooser")
    parser.add_option("--driver", dest="driver",
                      help="Runtime driver for jobs (local, or sge)",
                      type="string", default="local")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=20000)
    parser.add_option("--grid-seed", dest="grid_seed",
                      help="The seed used to initialize initial grid.",
                      type="int", default=1)
    parser.add_option("--config", dest="config_file",
                      help="Configuration file name.",
                      type="string", default="config.pb")
    parser.add_option("--run-job", dest="job",
                      help="Run a job in wrapper mode.",
                      type="string", default="")
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=3.0)

    (options, args) = parser.parse_args()

    return options, args


def main():
    (options, args) = parse_args()

    if options.job:
        job_runner(load_job(options.job))
        exit(0)

    expt_dir  = os.path.realpath(args[0])
    expt_name = os.path.basename(expt_dir)

    if not os.path.exists(expt_dir):
        log("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

    check_experiment_dirs(expt_dir)

    # Load up the chooser module.
    module  = importlib.import_module('chooser.' + options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)

    # Load up the job execution driver.
    module = importlib.import_module('driver.' + options.driver)
    driver = module.init()

    # Loop until we run out of jobs.
    while True:
        attempt_dispatch(expt_name, expt_dir, chooser, driver, options)
        # This is polling frequency. A higher frequency means that the algorithm
        # picks up results more quickly after they finish, but also significantly
        # increases overhead.
        time.sleep(options.polling_time)


# TODO:
#  * move check_pending_jobs out of ExperimentGrid, and implement two simple
#  driver classes to handle local execution and SGE execution.
#  * take cmdline engine arg into account, and submit job accordingly

def attempt_dispatch(expt_name, expt_dir, chooser, driver, options):
    log("\n")

    expt_file = os.path.join(expt_dir, options.config_file)
    expt      = load_expt(expt_file)

    # Build the experiment grid.
    expt_grid = ExperimentGrid(expt_dir,
                               expt.variable,
                               options.grid_size,
                               options.grid_seed)

    # Print out the current best function value.
    best_val, best_job = expt_grid.get_best()
    if best_job >= 0:
        log("Current best: %f (job %d)\n" % (best_val, best_job))
    else:
        log("Current best: No results returned yet.\n")

    # Gets you everything - NaN for unknown values & durations.
    grid, values, durations = expt_grid.get_grid()

    # Returns lists of indices.
    candidates = expt_grid.get_candidates()
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()

    n_candidates = candidates.shape[0]
    n_pending    = pending.shape[0]
    n_complete   = complete.shape[0]
    log("%d candidates   %d pending   %d complete\n" %
        (n_candidates, n_pending, n_complete))

    # Verify that pending jobs are actually running, and add them back to the
    # candidate set if they have crashed or gotten lost.
    for job_id in pending:
        proc_id = expt_grid.get_proc_id(job_id)
        if not driver.is_proc_alive(job_id, proc_id):
            log("Set job %d back to pending status.\n" % (job_id))
            expt_grid.set_candidate(job_id)

    # Track the time series of optimization.
    write_trace(expt_dir, best_val, best_job, n_candidates, n_pending, n_complete)

    # Print out the best job results
    write_best_job(expt_dir, best_val, best_job, expt_grid)

    if n_complete >= options.max_finished_jobs:
        log("Maximum number of finished jobs (%d) reached."
                         "Exiting\n" % options.max_finished_jobs)
        sys.exit(0)

    if n_candidates == 0:
        log("There are no candidates left.  Exiting.\n")
        sys.exit(0)

    if n_pending >= options.max_concurrent:
        log("Maximum number of jobs (%d) pending.\n" % (options.max_concurrent))

    else:

        # start a bunch of candidate jobs if possible
        #for i in range(min(options.max_concurrent - n_pending, n_candidates)):

        # Ask the chooser to pick the next candidate
        job_id = chooser.next(grid, values, durations, candidates, pending, complete)

        # If the job_id is a tuple, then the chooser picked a new job.
        # We have to add this to our grid
        if isinstance(job_id, tuple):
            (job_id, candidate) = job_id
            job_id = expt_grid.add_to_grid(candidate)

        log("Selected job %d from the grid... " % (job_id))

        # Convert this back into an interpretable job and add metadata.
        job = Job()
        job.id        = job_id
        job.expt_dir  = expt_dir
        job.name      = expt.name
        job.language  = expt.language
        job.status    = 'submitted'
        job.submit_t  = int(time.time())
        job.param.extend(expt_grid.get_params(job_id))

        save_job(job)
        pid = driver.submit_job(job)
        if pid:
            log("submitted - pid = %d\n" % (pid))
            expt_grid.set_submitted(job_id, pid)
        else:
            log("Failed to submit!\n")
            log("Deleting job file.\n")
            os.unlink(job_file_for(job))

    return


def write_trace(expt_dir, best_val, best_job,
                n_candidates, n_pending, n_complete):
    '''Append current experiment state to trace file.'''
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
    trace_fh.write("%d,%f,%d,%d,%d,%d\n"
                   % (time.time(), best_val, best_job,
                      n_candidates, n_pending, n_complete))
    trace_fh.close()


def write_best_job(expt_dir, best_val, best_job, expt_grid):
    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
    best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" %
                      (best_val, best_job))
    for best_params in expt_grid.get_params(best_job):
        best_job_fh.write(str(best_params) + '\n')
    best_job_fh.close()


def check_experiment_dirs(expt_dir):
    '''Make output and jobs sub directories.'''

    output_subdir = os.path.join(expt_dir, 'output')
    check_dir(output_subdir)

    job_subdir = os.path.join(expt_dir, 'jobs')
    check_dir(job_subdir)


def submit_job(job):
    name = "%s-%08d" % (job.name, job.id)

    # Submit the job.
    locker = Locker()
    locker.unlock(grid_for(job))
    proc = multiprocessing.Process(target=job_runner, args=[job])
    proc.start()

    if not proc.is_alive():
        log("Failed to submit job or job crashed "
                         "with return code %d !\n" % proc.exitcode)
        log("Deleting job file.\n")
        os.unlink(job_file_for(job))
        return None
    else:
        log("Submitted job as process: %d\n" % proc.pid)

    return proc


if __name__ == '__main__':
    main()
