import os
import multiprocessing

from dispatch import DispatchDriver
from helpers  import *
from runner   import job_runner


class LocalDriver(DispatchDriver):
    def submit_job(self, job):
       name = "%s-%08d" % (job.name, job.id)

       # TODO: figure out if this is necessary....
       #locker = Locker()
       #locker.unlock(grid_for(job))

       proc = multiprocessing.Process(target=job_runner, args=[job])
       proc.start()

       if proc.is_alive():
           return proc.pid
       else:
           return None


    def is_proc_alive(self, job_id, proc_id):
        try:
            # Send an alive signal to proc (note this could kill it in windows)
            os.kill(proc_id, 0)
        except OSError:
            return False

        return True

def init():
    print "local driver!"
    return LocalDriver()
