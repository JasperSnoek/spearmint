import os
import sys
import subprocess
import tempfile

from google.protobuf import text_format
from spearmint_pb2   import *


def log(*args):
    '''Write a msg to stderr.'''
    for v in args:
        sys.stderr.write(str(v))
    sys.stderr.write("\n")


def sh(cmd):
    '''Run a shell command (blocking until completion).'''
    subprocess.check_call(cmd, shell=True)


def redirect_output(path):
    '''Redirect stdout and stderr to a file.'''
    outfile    = open(path, 'a')
    sys.stdout = outfile
    sys.stderr = outfile


def check_dir(path):
    '''Create a directory if it doesn't exist.'''
    if not os.path.exists(path):
        os.mkdir(path)


def grid_for(job):
    return os.path.join(job.expt_dir, 'expt-grid.pkl')



def file_write_safe(path, data):
    '''Write data to a temporary file, then move to the destination path.'''
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fh.write(data)
    fh.close()
    cmd = 'mv "%s" "%s"' % (fh.name, path)
    sh(cmd)


def save_experiment(filename, expt):
    file_write_safe(filename, text_format.MessageToString(expt))


def load_experiment(filename):
    fh = open(filename, 'rb')
    expt = Experiment()
    text_format.Merge(fh.read(), expt)
    fh.close()
    return expt


def job_output_file(job):
    return os.path.join(job.expt_dir, 'output', '%08d.out' % (job.id))


def job_file_for(job):
    '''Get the path to the job file corresponding to a job object.'''
    return os.path.join(job.expt_dir, 'jobs', '%08d.pb' % (job.id))


def save_job(job):
    filename = job_file_for(job)
    file_write_safe(filename, job.SerializeToString())


def load_job(filename):
    fh = open(filename, 'rb')
    job = Job()
    job.ParseFromString(fh.read())
    fh.close()
    return job

