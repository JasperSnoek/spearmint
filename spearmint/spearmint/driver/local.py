import os
import multiprocessing

from dispatch import DispatchDriver
from helpers  import *
from runner   import job_runner
from Locker   import Locker

class LocalDriver(DispatchDriver):
    def submit_job(self, job):
       '''Submit a job for local execution.'''

       name = "%s-%08d" % (job.name, job.id)

       # TODO: figure out if this is necessary....
       locker = Locker()
       locker.unlock(grid_for(job))

       proc = multiprocessing.Process(target=job_runner, args=[job])
       proc.start()

       if proc.is_alive():
           log("Submitted job as process: %d" % proc.pid)
           return proc.pid
       else:
           log("Failed to submit job or job crashed "
               "with return code %d !" % proc.exitcode)
           log("Deleting job file.")
           os.unlink(job_file_for(job))
           return None


    def is_proc_alive(self, job_id, proc_id):
        try:
            # Send an alive signal to proc (note this could kill it in windows)
            os.kill(proc_id, 0)
        except OSError:
            return False

        return True


def init():
    return LocalDriver()
