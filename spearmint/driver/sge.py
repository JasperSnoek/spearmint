import subprocess
import drmaa

from dispatch import DispatchDriver
from helpers  import *


SGE_RUN_SCRIPT = '''
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
exec python2.7 spearmint.py --run-job "%s"

'''


class SGEDriver(DispatchDriver):
    def submit_job(name, output_file, modules, job_file, working_dir):

        sge_script = SGE_RUN_SCRIPT % (name, output_file, output_file, working_dir, " ".join(modules), job_file)

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
            return int(match.group(1))
        else:
            return None, output


    def is_proc_alive(sgeid):
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

        elif status in [drmaa.JobState.QUEUED_ACTIVE, drmaa.JobState.RUNNING]:
            pass # Good shape.

        elif status in [drmaa.JobState.SYSTEM_ON_HOLD,
                        drmaa.JobState.USER_ON_HOLD,
                        drmaa.JobState.USER_SYSTEM_ON_HOLD,
                        drmaa.JobState.SYSTEM_SUSPENDED,
                        drmaa.JobState.USER_SUSPENDED]:
            log("Job %d (%d) is held or suspended.\n" % (job_id, sgeid))
            reset_job = True

        elif status == drmaa.JobState.DONE:
            log("Job %d (%d) complete but not yet updated.\n" % (job_id, sgeid))

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

        s.exit()
        return True


def init():
    return SGEDriver()
