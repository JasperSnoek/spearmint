import sys
import os
import traceback

from spearmint_pb2   import *
from ExperimentGrid  import *
from helpers         import *


# System dependent modules
DEFAULT_MODULES = [ 'packages/epd/7.1-2',
                    'packages/matlab/r2011b',
                    'mpi/openmpi/1.2.8/intel',
                    'libraries/mkl/10.0',
                    'packages/cuda/4.0',
                    ]

MCR_LOCATION = "/home/matlab/v715" # hack


def job_runner(job):
    '''This fn runs in a new process.  Now we are going to do a little
    bookkeeping and then spin off the actual job that does whatever it is we're
    trying to achieve.'''

    redirect_output(job_output_file(job))
    log("Running in wrapper mode for '%s'\n" % (job.id))

    ExperimentGrid.job_running(job.expt_dir, job.id)

    # Update metadata and save the job file, which will be read by the job wrappers.
    job.start_t = int(time.time())
    job.status  = 'running'
    save_job(job)

    success    = False
    start_time = time.time()

    try:
        if job.language == MATLAB:   run_matlab_job(job)
        elif job.language == PYTHON: run_python_job(job)
        elif job.language == SHELL:  run_torch_job(job)
        elif job.language == MCR:    run_mcr_job(job)
        else:
            raise Exception("That function type has not been implemented.")

        success = True
    except:
        log("-" * 40)
        log("Problem running the job:")
        log(sys.exc_info())
        log(traceback.print_exc(limit=1000))
        log("-" * 40)

    end_time = time.time()
    duration = end_time - start_time

    # The job output is written back to the job file, so we read it back in to
    # get the results.
    job_file = job_file_for(job)
    job      = load_job(job_file)

    log("Job file reloaded.")

    if not job.HasField("value"):
        log("Could not find value in output file.")
        success = False

    if success:
        log("Completed successfully in %0.2f seconds. [%f]"
                         % (duration, job.value))

        # Update the status for this job.
        ExperimentGrid.job_complete(job.expt_dir, job.id,
                                    job.value, duration)
        job.status = 'complete'
    else:
        log("Job failed in %0.2f seconds." % (duration))

        # Update the experiment status for this job.
        ExperimentGrid.job_broken(job.expt_dir, job.id)
        job.status = 'broken'

    job.end_t    = int(time.time())
    job.duration = duration

    save_job(job)


def run_matlab_job(job):
    '''Run it as a Matlab function.'''

    log("Running matlab job.")

    job_file      = job_file_for(job)
    function_call = "matlab_wrapper('%s'),quit;" % (job_file)
    matlab_cmd    = ('matlab -nosplash -nodesktop -r "%s"' %
                     (function_call))
    log(matlab_cmd)
    sh(matlab_cmd)


# TODO: change this function to be more flexible when running python jobs
# regarding the python path, experiment directory, etc...
def run_python_job(job):
    '''Run a Python function.'''

    log("Running python job.\n")

    # Add experiment directory to the system path.
    sys.path.append(os.path.realpath(job.expt_dir))

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

    # Store the result.
    job.value = result
    save_job(job)


def run_torch_job(job):
    '''Run a torch based job.'''

    params = {}
    for param in job.param:
        dbl_vals = param.dbl_val._values
        int_vals = param.int_val._values
        str_vals = param.str_val._values

        if len(dbl_vals) > 0:
            params[param.name] = dbl_vals
        elif len(int_vals) > 0:
            params[param.name] = int_vals
        elif len(str_vals) > 0:
            params[param.name] = str_vals
        else:
            raise Exception("Unknown parameter type.")

    #TODO: this passes args correctly for experiment utils, but we need to
    # figure out how to get the result back out when the experiment completes.

    param_str = ""
    for pname, pval in params.iteritems():
        if len(pval) == 1:
            pval = str(pval[0])
        else:
            pval = ','.join([str(v) for v in pval])

        param_str += "-" + pname + " " + pval + " "

    cmd = "./%s %s" % (job.name, param_str)
    log("Executing command: %s\n" % (cmd))
    sh(cmd)


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


