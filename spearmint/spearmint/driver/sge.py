import os
import sys
import re
import subprocess
import drmaa

from dispatch import DispatchDriver
from helpers  import *


# TODO: figure out if these modules are necessary, or if they can be handled in
# the matlab runner or a user script...

# System dependent modules
# Note these are specific to the Harvard configuration
DEFAULT_MODULES = [ 'packages/epd/7.1-2',
                    'packages/matlab/r2011b',
                    'mpi/openmpi/1.2.8/intel',
                    'libraries/mkl/10.0',
                    'packages/cuda/4.0',
                    ]

# Removed from SGE script...
# Load matlab modules
#module load %s

class SGEDriver(DispatchDriver):

    def submit_job(self, job):
        output_file = job_output_file(job)
        job_file    = job_file_for(job)
        modules     = " ".join(DEFAULT_MODULES)
        mint_path   = sys.argv[0]
        sge_script  = 'python %s --run-job "%s" .' % (mint_path, job_file)

        qsub_cmd    = ['qsub', '-S', '/bin/bash',
                       '-N', "%s-%d" % (job.name, job.id),
                       '-e', output_file,
                       '-o', output_file,
                       '-j', 'y',
                      ]

        process = subprocess.Popen(" ".join(qsub_cmd),
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   shell=True)
        output = process.communicate(input=sge_script)[0]
        process.stdin.close()

        # Parse out the job id.
        match = re.search(r'Your job (\d+)', output)

        if match:
            return int(match.group(1))
        else:
            return None, output


    def is_proc_alive(self, job_id, sgeid):
        try:
            s = drmaa.Session()
            s.initialize()

            reset_job = False

            try:
                status = s.jobStatus(str(sgeid))
            except:
                log("EXC: %s\n" % (str(sys.exc_info()[0])))
                log("Could not find SGE id for job %d (%d)\n" % (job_id, sgeid))
                status = -1
                reset_job = True

            if status == drmaa.JobState.UNDETERMINED:
                log("Job %d (%d) in undetermined state.\n" % (job_id, sgeid))
                reset_job = True

            elif status == drmaa.JobState.QUEUED_ACTIVE:
                log("Job %d (%d) waiting in queue.\n" % (job_id, sgeid))

            elif status == drmaa.JobState.RUNNING:
                log("Job %d (%d) is running.\n" % (job_id, sgeid))

            elif status in [drmaa.JobState.SYSTEM_ON_HOLD,
                            drmaa.JobState.USER_ON_HOLD,
                            drmaa.JobState.USER_SYSTEM_ON_HOLD,
                            drmaa.JobState.SYSTEM_SUSPENDED,
                            drmaa.JobState.USER_SUSPENDED]:
                log("Job %d (%d) is held or suspended.\n" % (job_id, sgeid))
                reset_job = True

            elif status == drmaa.JobState.DONE:
                log("Job %d (%d) is finished.\n" % (job_id, sgeid))

            elif status == drmaa.JobState.FAILED:
                log("Job %d (%d) failed.\n" % (job_id, sgeid))
                reset_job = True

            if reset_job:

                try:
                    # Kill the job.
                    s.control(str(sgeid), drmaa.JobControlAction.TERMINATE)
                    log("Killed SGE job %d.\n" % (sgeid))
                except:
                    log("Failed to kill SGE job %d.\n" % (sgeid))

                return False
            else:
                return True

        finally:
            s.exit()


def init():
    return SGEDriver()

