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
