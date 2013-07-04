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
import time
import imp
import os
import re
import Locker

from google.protobuf import text_format
from spearmint_pb2   import *
from ExperimentGrid  import *

# System dependent modules
MCR_LOCATION = "/home/matlab/v715" # hack


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
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=3.0)

    (options, args) = parser.parse_args()

    return options, args


def main():
    (options, args) = parse_args()

    expt_dir  = os.path.realpath(args[0])
    work_dir  = os.path.realpath('.')
    expt_name = os.path.basename(expt_dir)

    if not os.path.exists(expt_dir):
        log("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

    check_experiment_dirs(expt_dir)

    # Load up the chooser module.
    module  = __import__(options.chooser_module)
    chooser = module.init(expt_dir, options.chooser_args)

    # Loop until we run out of jobs.
    while True:
        attempt_dispatch(expt_name, expt_dir, work_dir, chooser, options)
        # This is polling frequency. A higher frequency means that the algorithm
        # picks up results more quickly after they finish, but also significantly
        # increases overhead.
        time.sleep(options.polling_time)


def attempt_dispatch(expt_name, expt_dir, work_dir, chooser, options):
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
    log("%d candidates   %d pending   %d complete\n" %
                     (candidates.shape[0], pending.shape[0], complete.shape[0]))

    # Verify that pending jobs are actually running.
    expt_grid.check_pending_jobs()

    # Track the time series of optimization.
    write_trace(expt_dir, best_val, best_job,
                candidates.shape[0], pending.shape[0], complete.shape[0])

    # Print out the best job results
    write_best_job(expt_dir, best_val, best_job, expt_grid)

    if complete.shape[0] >= options.max_finished_jobs:
        log("Maximum number of finished jobs (%d) reached."
                         "Exiting\n" % options.max_finished_jobs)
        sys.exit(0)

    if candidates.shape[0] == 0:
        log("There are no candidates left.  Exiting.\n")
        sys.exit(0)

    if pending.shape[0] >= options.max_concurrent:
        log("Maximum number of jobs (%d) pending.\n"
                         % (options.max_concurrent))
        return

    # Ask the chooser to actually pick one.
    job_id = chooser.next(grid, values, durations, candidates, pending, complete)

    # If the job_id is a tuple, then the chooser picked a new job.
    # We have to add this to our grid
    if isinstance(job_id, tuple):
        (job_id, candidate) = job_id
        job_id = expt_grid.add_to_grid(candidate)

    log("Selected job %d from the grid.\n" % (job_id))

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
    process = submit_job(job, work_dir)
    expt_grid.set_submitted(job_id, process.pid)

    return



def job_runner(job):
    '''This fn runs in a new process.  Now we are going to do a little
    bookkeeping and then spin off the actual job that does whatever it is we're
    trying to achieve.'''

    redirect_output(job_output_file(job))
    log("Running in wrapper mode for '%s'\n" % (job.id))

    ExperimentGrid.job_running(job.expt_dir, job.id)

    # Update metadata and save the job file, which will be read by the job
    # wrappers.
    job.start_t = int(time.time())
    job.status  = 'running'
    save_job(job)

    success    = False
    start_time = time.time()

    try:
        if job.language == MATLAB:   run_matlab_job(job)
        elif job.language == PYTHON: run_python_job(job)
        elif job.language == SHELL:  run_shell_job(job)
        elif job.language == MCR:    run_mcr_job(job)
        else:
            raise Exception("That function type has not been implemented.")

        success = True
    except:
        log("-" * 40 + "\n")
        log("Problem executing the function:\n")
        print sys.exc_info()

    end_time = time.time()
    duration = end_time - start_time

    # The job output is written back to the job file, so we read it back in to
    # get the results.
    job_file = job_file_for(job)
    job      = load_job(job_file)

    log("Job file reloaded.\n")

    if not job.HasField("value"):
        log("Could not find value in output file.\n")
        success = False

    if success:
        log("Completed successfully in %0.2f seconds. [%f]\n"
                         % (duration, job.value))

        # Update the status for this job.
        ExperimentGrid.job_complete(job.expt_dir, job.id,
                                    job.value, duration)
        job.status = 'complete'
    else:
        log("Job failed in %0.2f seconds.\n" % (duration))

        # Update the experiment status for this job.
        ExperimentGrid.job_broken(job.expt_dir, job.id)
        job.status = 'broken'

    job.end_t    = int(time.time())
    job.duration = duration

    save_job(job)



def run_matlab_job(job):
    '''Run it as a Matlab function.'''

    log("Running matlab job.\n")

    job_file      = job_file_for(job)
    function_call = "matlab_wrapper('%s'),quit;" % (job_file)
    matlab_cmd    = ('matlab -nosplash -nodesktop -r "%s"' %
                     (function_call))
    log(matlab_cmd + "\n")
    sh(matlab_cmd)


# TODO: change this function to be more flexible when running python jobs
# regarding the python path, experiment directory, etc...
def run_python_job(job):
    '''Run a Python function.'''

    log("Running python job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job.expt_dir))

    # Change into the directory.
    os.chdir(job.expt_dir)
    log("Changed into dir %s\n" % (os.getcwd()))

    # Convert the PB object into useful parameters.
    params = {}
    for param in job.param:
        dbl_vals = param.dbl_val._values
        int_vals = param.int_val._values
        str_vals = param.str_val._values

        if len(dbl_vals) > 0:
            params[param.name] = np.array(dbl_vals)
        elif len(int_vals) > 0:
            params[param.name] = np.array(int_vals, dtype=int)
        elif len(str_vals) > 0:
            params[param.name] = str_vals
        else:
            raise Exception("Unknown parameter type.")

    # Load up this module and run
    module  = __import__(job.name)
    result = module.main(job.id, params)

    log("Got result %f\n" % (result))

    # Change back out.
    os.chdir('..')

    # Store the result.
    job.value = result
    save_job(job)

def run_shell_job(job):
    '''Run a shell based job.'''

    log("Running shell job.\n")

    # Change into the directory.
    os.chdir(job.expt_dir)

    cmd      = './%s %s' % (job.name, job_file_for(job))
    log("Executing command '%s'\n" % (cmd))

    sh(cmd)


def run_mcr_job(job):
    '''Run a compiled Matlab job.'''

    log("Running a compiled Matlab job.\n")

    # Change into the directory.
    os.chdir(job.expt_dir)

    if os.environ.has_key('MATLAB'):
        mcr_loc = os.environ['MATLAB']
    else:
        mcr_loc = MCR_LOCATION

    cmd = './run_%s.sh %s %s' % (job.name, mcr_loc, job_file_for(job))
    log("Executing command '%s'\n" % (cmd))
    sh(cmd)


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


def load_expt(filename):
    fh = open(filename, 'rb')
    expt = Experiment()
    text_format.Merge(fh.read(), expt)
    fh.close()
    return expt

def load_job(filename):
    fh = open(filename, 'rb')
    job = Job()
    #text_format.Merge(fh.read(), job)
    job.ParseFromString(fh.read())
    fh.close()
    return job

def save_expt(filename, expt):
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fh.write(text_format.MessageToString(expt))
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, filename)
    sh(cmd)

def check_dir(path):
    '''Create a directory if it doesn't exist.'''
    if not os.path.exists(path):
        os.mkdir(path)

def check_experiment_dirs(expt_dir):
    '''Make output and jobs sub directories.'''

    output_subdir = os.path.join(expt_dir, 'output')
    check_dir(output_subdir)

    job_subdir = os.path.join(expt_dir, 'jobs')
    check_dir(job_subdir)


def job_file_for(job):
    return os.path.join(job.expt_dir, 'jobs', '%08d.pb' % (job.id))


def job_output_file(job):
    return os.path.join(job.expt_dir, 'output', '%08d.out' % (job.id))


def redirect_output(path):
    outfile    = open(path, 'w')
    sys.stdout = outfile
    sys.stderr = outfile


def save_job(job):
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fh.write(job.SerializeToString())
    fh.close()

    job_file = job_file_for(job)
    cmd = 'mv "%s" "%s"' % (fh.name, job_file)
    sh(cmd)


def submit_job(job, working_dir):
    name = "%s-%08d" % (job.name, job.id)

    #output_file = open(output_file, 'w')

    # Submit the job.
    locker = Locker()
    locker.unlock(working_dir + '/expt-grid.pkl')
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


def log(msg):
    sys.stderr.write(msg)


def sh(cmd):
    subprocess.check_call(cmd, shell=True)


if __name__ == '__main__':
    main()
