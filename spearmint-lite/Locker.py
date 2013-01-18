import os
import sys
import time

def safe_delete(filename):
    cmd  = 'mv "%s" "%s.delete" && rm "%s.delete"' % (filename, filename, 
                                                      filename)
    fail = os.system(cmd)
    return not fail

class Locker:

    def __init__(self):
        self.locks = {}

    def __del__(self):
        for filename in self.locks.keys():
            self.locks[filename] = 1
            self.unlock(filename)

    def lock(self, filename):
        if self.locks.has_key(filename):
            self.locks[filename] += 1
            return True
        else:
            cmd = 'ln -s /dev/null "%s.lock" 2> /dev/null' % (filename)
            fail = os.system(cmd)
            if not fail:
                self.locks[filename] = 1
            return not fail

    def lock(self, filename):
        if self.locks.has_key(filename):
            self.locks[filename] += 1
            return True
        else:
            cmd = 'ln -s /dev/null "%s.lock" 2> /dev/null' % (filename)
            fail = os.system(cmd)
            if not fail:
                self.locks[filename] = 1
            return not fail

    def unlock(self, filename):
        if not self.locks.has_key(filename):
            sys.stderr.write("Trying to unlock not-locked file %s.\n" % 
                             (filename))
            return True
        if self.locks[filename] == 1:
            success = safe_delete('%s.lock' % (filename))
            if not success:
                sys.stderr.write("Could not unlock file: %s.\n" % (filename))
            del self.locks[filename]
            return success
        else:
            self.locks[filename] -= 1
            return True
            
    def lock_wait(self, filename):
        while not self.lock(filename):
          time.sleep(0.01)
