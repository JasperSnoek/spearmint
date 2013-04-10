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
    parser.add_option("--wrapper", dest="wrapper",
                      help="Run in job-wrapper mode.",
                      action="store_true")
    parser.add_option("--polling-time", dest="polling_time",
                      help="The time in-between successive polls for results.",
                      type="float", default=3.0)

    (options, args) = parser.parse_args()

    if options.wrapper:
        # Possibly run in job wrapper mode.
        main_wrapper(options, args)

    else:
        # Otherwise run in controller mode.
        main_controller(options, args)
    
##############################################################################
##############################################################################
def main_wrapper(options, args):
    sys.stderr.write("Running in wrapper mode for '%s'\n" % (args[0]))

    # This happens when the job is actually executing.  Now we are
    # going to do a little bookkeeping and then spin off the actual
    # job that does whatever it is we're trying to achieve.

    # Load in the protocol buffer spec for this job and experiment.
    job_file = args[0]
    job      = load_job(job_file)

    ExperimentGrid.job_running(job.expt_dir, job.id)
    
    # Update metadata.
    job.start_t = int(time.time())
    job.status  = 'running'
    save_job(job_file, job)

    ##########################################################################
    success    = False
    start_time = time.time()

    try:
        if job.language == MATLAB:
            # Run it as a Matlab function.
            function_call = "matlab_wrapper('%s'),quit;" % (job_file)
            matlab_cmd    = ('matlab -nosplash -nodesktop -r "%s"' % 
                             (function_call))
            sys.stderr.write(matlab_cmd + "\n")
            subprocess.check_call(matlab_cmd, shell=True)

        elif job.language == PYTHON:
            # Run a Python function
            sys.stderr.write("Running python job.\n")

            # Add directory to the system path.
            sys.path.append(os.path.realpath(job.expt_dir))

            # Change into the directory.
            os.chdir(job.expt_dir)
            sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

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

            sys.stderr.write("Got result %f\n" % (result))

            # Change back out.
            os.chdir('..')

            # Store the result.
            job.value = result
            save_job(job_file, job)

        elif job.language == SHELL:
            # Change into the directory.
            os.chdir(job.expt_dir)

            cmd = './%s %s' % (job.name, job_file)
            sys.stderr.write("Executing command '%s'\n" % (cmd))

            subprocess.check_call(cmd, shell=True)

        elif job.language == MCR:

            # Change into the directory.
            os.chdir(job.expt_dir)

            if os.environ.has_key('MATLAB'):
                mcr_loc = os.environ['MATLAB']
            else:
                mcr_loc = MCR_LOCATION

            cmd = './run_%s.sh %s %s' % (job.name, mcr_loc, job_file)
            sys.stderr.write("Executing command '%s'\n" % (cmd))
            subprocess.check_call(cmd, shell=True)

        else:
            raise Exception("That function type has not been implemented.")

        success = True
    except:
        sys.stderr.write("Problem executing the function\n")
        print sys.exc_info()
        
    end_time = time.time()
    duration = end_time - start_time
    ##########################################################################

    job = load_job(job_file)
    sys.stderr.write("Job file reloaded.\n")

    if not job.HasField("value"):
        sys.stderr.write("Could not find value in output file.\n")
        success = False

    if success:
        sys.stderr.write("Completed successfully in %0.2f seconds. [%f]\n" 
                         % (duration, job.value))

        # Update the status for this job.
        ExperimentGrid.job_complete(job.expt_dir, job.id,
                                    job.value, duration)
    
        # Update metadata.
        job.end_t    = int(time.time())
        job.status   = 'complete'
        job.duration = duration

    else:
        sys.stderr.write("Job failed in %0.2f seconds.\n" % (duration))

        # Update the status for this job.
        ExperimentGrid.job_broken(job.expt_dir, job.id)
    
        # Update metadata.
        job.end_t    = int(time.time())
        job.status   = 'broken'
        job.duration = duration

    save_job(job_file, job)

##############################################################################
##############################################################################
def main_controller(options, args):

    expt_dir  = os.path.realpath(args[0])
    work_dir  = os.path.realpath('.')
    expt_name = os.path.basename(expt_dir)

    if not os.path.exists(expt_dir):
        sys.stderr.write("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

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
    sys.stderr.write("\n")
    
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
        sys.stderr.write("Current best: %f (job %d)\n" % (best_val, best_job))
    else:
        sys.stderr.write("Current best: No results returned yet.\n")

    # Gets you everything - NaN for unknown values & durations.
    grid, values, durations = expt_grid.get_grid()
    
    # Returns lists of indices.
    candidates = expt_grid.get_candidates()
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()
    sys.stderr.write("%d candidates   %d pending   %d complete\n" % 
                     (candidates.shape[0], pending.shape[0], complete.shape[0]))
      
    # Verify that pending jobs are actually running.
    for job_id in pending:
        sgeid = expt_grid.get_sgeid(job_id)
        reset_job = False
        
        try:
            # Send an alive signal to proc (note this could kill it in windows)
            os.kill(sgeid, 0)
        except OSError:
            # Job is no longer running but still in the candidate list. Assume it crashed out.
            expt_grid.set_candidate(job_id)

    # Track the time series of optimization.
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
    trace_fh.write("%d,%f,%d,%d,%d,%d\n"
                   % (time.time(), best_val, best_job,
                      candidates.shape[0], pending.shape[0], complete.shape[0]))
    trace_fh.close()

    # Print out the best job results
    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'w')
    best_job_fh.write("Best result: %f\nJob-id: %d\nParameters: \n" % 
                      (best_val, best_job))    
    for best_params in expt_grid.get_params(best_job):
        best_job_fh.write(str(best_params) + '\n')
    best_job_fh.close()

    if complete.shape[0] >= options.max_finished_jobs:
        sys.stderr.write("Maximum number of finished jobs (%d) reached."
                         "Exiting\n" % options.max_finished_jobs)
        sys.exit(0)

    if candidates.shape[0] == 0:
        sys.stderr.write("There are no candidates left.  Exiting.\n")
        sys.exit(0)

    if pending.shape[0] >= options.max_concurrent:
        sys.stderr.write("Maximum number of jobs (%d) pending.\n"
                         % (options.max_concurrent))
        return

    # Ask the chooser to actually pick one.
    job_id = chooser.next(grid, values, durations, candidates, pending,
                          complete)

    # If the job_id is a tuple, then the chooser picked a new job.
    # We have to add this to our grid
    if isinstance(job_id, tuple):
        (job_id, candidate) = job_id
        job_id = expt_grid.add_to_grid(candidate)

    sys.stderr.write("Selected job %d from the grid.\n" % (job_id))

    # Convert this back into an interpretable job and add metadata.
    job = Job()
    job.id        = job_id
    job.expt_dir  = expt_dir
    job.name      = expt.name
    job.language  = expt.language
    job.status    = 'submitted'
    job.submit_t  = int(time.time())
    job.param.extend(expt_grid.get_params(job_id))

    # Make sure we have a job subdirectory.
    job_subdir = os.path.join(expt_dir, 'jobs')
    if not os.path.exists(job_subdir):
        os.mkdir(job_subdir)

    # Name this job file.
    job_file = os.path.join(job_subdir,
                            '%08d.pb' % (job_id))

    # Store the job file.
    save_job(job_file, job)

    # Make sure there is a directory for output.
    output_subdir = os.path.join(expt_dir, 'output')
    if not os.path.exists(output_subdir):
        os.mkdir(output_subdir)
    output_file = os.path.join(output_subdir,
                               '%08d.out' % (job_id))

    process = job_submit("%s-%08d" % (expt_name, job_id),
                         output_file,
                         job_file, work_dir)
    process.poll()
    if process.returncode is not None and process.returncode < 0:
        sys.stderr.write("Failed to submit job or job crashed "
                         "with return code %d !\n" % process.returncode)
        sys.stderr.write("Deleting job file.\n")
        os.unlink(job_file)
        return
    else:
        sys.stderr.write("Submitted job as process: %d\n" % process.pid)

    # Now, update the experiment status to submitted.
    expt_grid.set_submitted(job_id, process.pid)

    return

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
    subprocess.check_call(cmd, shell=True)

def save_job(filename, job):
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    #fh.write(text_format.MessageToString(job))
    fh.write(job.SerializeToString())
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, filename)
    subprocess.check_call(cmd, shell=True)

def job_submit(name, output_file, job_file, working_dir):

    cmd = ('''python spearmint_sync.py --wrapper "%s" > %s''' % 
           (job_file, output_file))
    output_file = open(output_file, 'w')

    # Submit the job.
    locker = Locker()
    locker.unlock(working_dir + '/expt-grid.pkl')
    process = subprocess.Popen(cmd,
                               stdout=output_file,
                               stderr=output_file, shell=True)

    return process

if __name__ == '__main__':
    main()
