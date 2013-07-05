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

from google.protobuf import text_format
from spearmint_pb2   import *
from ExperimentGrid  import *

DEFAULT_MODULES = [ 'packages/epd/7.1-2',
                    'packages/matlab/r2011b',
                    'mpi/openmpi/1.2.8/intel',
                    'libraries/mkl/10.0',
                    'packages/cuda/4.0',
                    ]
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
                      type="int", default=1000)
    parser.add_option("--method", dest="chooser_module",
                      help="Method for choosing experiments.",
                      type="string", default="GPEIChooser")
    parser.add_option("--method-args", dest="chooser_args",
                      help="Arguments to pass to chooser module.",
                      type="string", default="")
    parser.add_option("--grid-size", dest="grid_size",
                      help="Number of experiments in initial grid.",
                      type="int", default=10000)
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

    # Load in the Protocol buffer spec for this job and experiment.
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
            matlab_cmd    = 'matlab -nosplash -nodesktop -r "%s"' % (function_call)
            sys.stderr.write(matlab_cmd + "\n")
            os.system(matlab_cmd)

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

            os.system(cmd)

        elif job.language == MCR:

            # Change into the directory.
            os.chdir(job.expt_dir)

            if os.environ.has_key('MATLAB'):
                mcr_loc = os.environ['MATLAB']
            else:
                mcr_loc = MCR_LOCATION

            cmd = './run_%s.sh %s %s' % (job.name, mcr_loc, job_file)
            sys.stderr.write("Executing command '%s'\n" % (cmd))
            os.system(cmd)

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
        sys.stderr.write("Cannot find experiment directory '%s'.  Aborting.\n" % (expt_dir))
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
    import drmaa

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
    sys.stderr.write("Current best: %f (job %d)\n" % (best_val, best_job))

    # Gets you everything - NaN for unknown values & durations.
    grid, values, durations = expt_grid.get_grid()

    # Returns lists of indices.
    candidates = expt_grid.get_candidates()
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()
    sys.stderr.write("%d candidates   %d pending   %d complete\n" %
                     (candidates.shape[0], pending.shape[0], complete.shape[0]))

    # Verify that pending jobs are actually running.
    s = drmaa.Session()
    s.initialize()
    for job_id in pending:
        sgeid = expt_grid.get_sgeid(job_id)
        reset_job = False

        try:
            status = s.jobStatus(str(sgeid))
        except:
            sys.stderr.write("EXC: %s\n" % (str(sys.exc_info()[0])))
            sys.stderr.write("Could not find SGE id for job %d (%d)\n" %
                             (job_id, sgeid))
            status = -1
            reset_job = True

        if status == drmaa.JobState.UNDETERMINED:
            sys.stderr.write("Job %d (%d) in undetermined state.\n" %
                             (job_id, sgeid))
            reset_job = True

        elif status in [drmaa.JobState.QUEUED_ACTIVE, drmaa.JobState.RUNNING]:
            pass # Good shape.

        elif status in [drmaa.JobState.SYSTEM_ON_HOLD,
                        drmaa.JobState.USER_ON_HOLD,
                        drmaa.JobState.USER_SYSTEM_ON_HOLD,
                        drmaa.JobState.SYSTEM_SUSPENDED,
                        drmaa.JobState.USER_SUSPENDED]:
            sys.stderr.write("Job %d (%d) is held or suspended.\n" %
                             (job_id, sgeid))
            reset_job = True

        elif status == drmaa.JobState.DONE:
            sys.stderr.write("Job %d (%d) complete but not yet updated.\n" %
                             (job_id, sgeid))

        elif status == drmaa.JobState.FAILED:
            sys.stderr.write("Job %d (%d) failed.\n" % (job_id, sgeid))
            reset_job = True

        if reset_job:

            try:
                # Kill the job.
                s.control(str(sgeid), drmaa.JobControlAction.TERMINATE)
                sys.stderr.write("Killed SGE job %d.\n" % (sgeid))
            except:
                sys.stderr.write("Failed to kill SGE job %d.\n" % (sgeid))

            # Set back to being a candidate state.
            expt_grid.set_candidate(job_id)
            sys.stderr.write("Set job %d back to pending status.\n" % (job_id))

    s.exit()

    # Track the time series of optimization.
    trace_fh = open(os.path.join(expt_dir, 'trace.csv'), 'a')
    trace_fh.write("%d,%f,%d,%d,%d,%d\n"
                   % (time.time(), best_val, best_job,
                      candidates.shape[0], pending.shape[0], complete.shape[0]))
    trace_fh.close()

    # Print out the best job results
    best_job_fh = open(os.path.join(expt_dir, 'best_job_and_result.txt'), 'a')
    best_job_fh.write("Best result: %f\n Job-id: %d\n Parameters: %s\n"
                      % (best_val, best_job, expt_grid.get_params(best_job)))
    best_job_fh.close()

    if complete.shape[0] >= options.max_finished_jobs:
        sys.stderr.write("Maximum number of finished jobs (%d) reached. "
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

    queue_id, msg = sge_submit("%s-%08d" % (expt_name, job_id),
                             output_file,
                             DEFAULT_MODULES,
                             job_file, work_dir)
    if queue_id is None:
        sys.stderr.write("Failed to submit job: %s" % (msg))
        sys.stderr.write("Deleting job file.\n")
        os.unlink(job_file)
        return
    else:
        sys.stderr.write("Submitted as job %d\n" % (queue_id))

    # Now, update the experiment status to submitted.
    expt_grid.set_submitted(job_id, queue_id)

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
    os.system(cmd)

def save_job(filename, job):
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    #fh.write(text_format.MessageToString(job))
    fh.write(job.SerializeToString())
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, filename)
    os.system(cmd)

def sge_submit(name, output_file, modules, job_file, working_dir):

    sge_script = '''
#!/bin/bash
#$ -S /bin/bash
#$ -N "%s"
#$ -j yes
#$ -e "%s"
#$ -o "%s"
#$ -wd "%s"

# Set up the environment
. /etc/profile
. ~/.profile

# Make sure we have various modules.
module load %s

# Spin off ourselves as a wrapper script.
exec python2.7 spearmint.py --wrapper "%s"

''' % (name, output_file, output_file, working_dir, " ".join(modules), job_file)

    # Submit the job.
    process = subprocess.Popen('qsub',
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=False)
    output = process.communicate(input=sge_script)[0]
    process.stdin.close()

    # Parse out the job id.
    match = re.search(r'Your job (\d+)', output)
    if match:
        return int(match.group(1)), output
    else:
        return None, output

if __name__ == '__main__':
    main()
