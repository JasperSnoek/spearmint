##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#                                                                                                                                                                                  
# This code is written for research and educational purposes only to
# supplement the paper entitled "Practical Bayesian Optimization of
# Machine Learning Algorithms" by Snoek, Larochelle and Adams Advances
# in Neural Information Processing Systems, 2012
#                                                                                                                                                                               
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#                                                                                                                                                                             
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#                                                                                                                                                                        
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
import os
import sys
import tempfile
import cPickle

import numpy        as np
import numpy.random as npr

from Locker        import *
from sobol_lib     import *

CANDIDATE_STATE = 0
SUBMITTED_STATE = 1
RUNNING_STATE   = 2
COMPLETE_STATE  = 3
BROKEN_STATE    = -1

class ExperimentGrid:

    @staticmethod
    def job_running(expt_dir, id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_running(id)

    @staticmethod
    def job_complete(expt_dir, id, value, duration):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_complete(id, value, duration)

    @staticmethod
    def job_broken(expt_dir, id):
        expt_grid = ExperimentGrid(expt_dir)
        expt_grid.set_broken(id)

    def __init__(self, expt_dir, variables=None, grid_size=None, grid_seed=1):
        self.expt_dir = expt_dir
        self.jobs_pkl = os.path.join(expt_dir, 'expt-grid.pkl')
        self.locker   = Locker()

        # Only one process at a time is allowed to have access to this.
        sys.stderr.write("Waiting to lock grid...")
        self.locker.lock_wait(self.jobs_pkl)
        sys.stderr.write("...acquired\n")

        # Does this exist already?
        if variables is not None and not os.path.exists(self.jobs_pkl):

            # Set up the grid for the first time.
            self.seed = grid_seed
            self.vmap   = GridMap(variables, grid_size)
            self.grid   = self.hypercube_grid(self.vmap.card(), grid_size)
            self.status = np.zeros(grid_size, dtype=int) + CANDIDATE_STATE
            self.values = np.zeros(grid_size) + np.nan
            self.durs   = np.zeros(grid_size) + np.nan
            self.sgeids = np.zeros(grid_size, dtype=int)

            # Save this out.
            self._save_jobs()
        else:

            # Load in from the pickle.
            self._load_jobs()

    def __del__(self):
        self._save_jobs()
        if self.locker.unlock(self.jobs_pkl):
            sys.stderr.write("Released lock on job grid.\n")
        else:
            raise Exception("Could not release lock on job grid.\n")

    def get_grid(self):
        return self.grid, self.values, self.durs

    def get_candidates(self):
        return np.nonzero(self.status == CANDIDATE_STATE)[0]

    def get_pending(self):
        return np.nonzero((self.status == SUBMITTED_STATE) | (self.status == RUNNING_STATE))[0]

    def get_complete(self):
        return np.nonzero(self.status == COMPLETE_STATE)[0]

    def get_broken(self):
        return np.nonzero(self.status == BROKEN_STATE)[0]

    def get_params(self, index):
        return self.vmap.get_params(self.grid[index,:])

    def get_best(self):
        finite = self.values[np.isfinite(self.values)]
        if len(finite) > 0:
            cur_min = np.min(finite)
            index   = np.nonzero(self.values==cur_min)[0][0]
            return cur_min, index
        else:
            return np.nan, -1

    def get_sgeid(self, id):
        return self.sgeids[id]

    def add_to_grid(self, candidate):
        # Set up the grid
        self.grid   = np.vstack((self.grid, candidate))
        self.status = np.append(self.status, np.zeros(1, dtype=int) + 
                                int(CANDIDATE_STATE))
        
        self.values = np.append(self.values, np.zeros(1)+np.nan)
        self.durs   = np.append(self.durs, np.zeros(1)+np.nan)
        self.sgeids = np.append(self.sgeids, np.zeros(1,dtype=int))

        # Save this out.
        self._save_jobs()
        return self.grid.shape[0]-1

    def set_candidate(self, id):
        self.status[id] = CANDIDATE_STATE
        self._save_jobs()

    def set_submitted(self, id, sgeid):
        self.status[id] = SUBMITTED_STATE
        self.sgeids[id] = sgeid
        self._save_jobs()

    def set_running(self, id):
        self.status[id] = RUNNING_STATE
        self._save_jobs()

    def set_complete(self, id, value, duration):
        self.status[id] = COMPLETE_STATE
        self.values[id] = value
        self.durs[id]   = duration
        self._save_jobs()

    def set_broken(self, id):
        self.status[id] = BROKEN_STATE
        self._save_jobs()

    def _load_jobs(self):
        fh   = open(self.jobs_pkl, 'r')
        jobs = cPickle.load(fh)
        fh.close()

        self.vmap   = jobs['vmap']
        self.grid   = jobs['grid']
        self.status = jobs['status']
        self.values = jobs['values']
        self.durs   = jobs['durs']
        self.sgeids = jobs['sgeids']

    def _save_jobs(self):

        # Write everything to a temporary file first.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'vmap'   : self.vmap,
                       'grid'   : self.grid,
                       'status' : self.status,
                       'values' : self.values,
                       'durs'   : self.durs,
                       'sgeids' : self.sgeids }, fh)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.jobs_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.
    
    def _hypercube_grid(self, dims, size):
        # Generate from a sobol sequence
        sobol_grid = np.transpose(i4_sobol_generate(dims,size,self.seed))
                
        return sobol_grid

class Parameter:
    def __init__(self):
        self.type = []
        self.name = []
        self.type = []
        self.min = []
        self.max = []
        self.options = []
        self.int_val = []
        self.dbl_val = []
        self.str_val = []

class GridMap:
    
    def __init__(self, variables, grid_size):
        self.variables   = []
        self.cardinality = 0

        # Count the total number of dimensions and roll into new format.
        for variable in variables:
            self.cardinality += variable['size']

            if variable['type'] == 'int':
                self.variables.append({ 'name' : variable['name'],
                                        'size' : variable['size'],
                                        'type' : 'int',
                                        'min'  : int(variable['min']),
                                        'max'  : int(variable['max'])})

            elif variable['type'] == 'float':
                self.variables.append({ 'name' : variable['name'],
                                        'size' : variable['size'],
                                        'type' : 'float',
                                        'min'  : float(variable['min']),
                                        'max'  : float(variable['max'])})

            elif variable['type'] == 'enum':
                self.variables.append({ 'name'    : variable['name'],
                                        'size'    : variable['size'],
                                        'type'    : 'enum',
                                        'options' : list(variable['options'])})
            else:
                raise Exception("Unknown parameter type.")
        sys.stderr.write("Optimizing over %d dimensions\n" % (self.cardinality))

    # Get a list of candidate experiments generated from a sobol sequence
    def hypercube_grid(self, size, seed):
        # Generate from a sobol sequence
        sobol_grid = np.transpose(i4_sobol_generate(self.cardinality,size,seed))
                
        return sobol_grid

    # Convert a variable to the unit hypercube
    # Takes a single variable encoded as a list, assuming the ordering is 
    # the same as specified in the configuration file
    def to_unit(self, v):
        unit = np.zeros(self.cardinality)
        index  = 0

        for variable in self.variables:
            #param.name = variable['name']
            if variable['type'] == 'int':
                for dd in xrange(variable['size']):
                    unit[index] = self._index_unmap(float(v.pop(0)) - variable['min'], (variable['max']-variable['min'])+1)
                    index += 1

            elif variable['type'] == 'float':
                for dd in xrange(variable['size']):
                    unit[index] = (float(v.pop(0)) - variable['min'])/(variable['max']-variable['min'])
                    index += 1

            elif variable['type'] == 'enum':
                for dd in xrange(variable['size']):
                    unit[index] = variable['options'].index(v.pop(0))
                    index += 1

            else:
                raise Exception("Unknown parameter type.")
            
        if (len(v) > 0):
            raise Exception("Too many variables passed to parser")
        return unit

    def unit_to_list(self, u):
        params = self.get_params(u)
        paramlist = []
        for p in params:
            if p.type == 'int':
                for v in p.int_val:
                    paramlist.append(v)
            if p.type == 'float':
                for v in p.dbl_val:
                    paramlist.append(v)
            if p.type == 'enum':
                for v in p.str_val:
                    paramlist.append(v)
        return paramlist
        
    def get_params(self, u):
        if u.shape[0] != self.cardinality:
            raise Exception("Hypercube dimensionality is incorrect.")

        params = []
        index  = 0
        for variable in self.variables:
            param = Parameter()
            
            param.name = variable['name']
            if variable['type'] == 'int':
                param.type = 'int'
                for dd in xrange(variable['size']):
                    param.int_val.append(variable['min'] + self._index_map(u[index], variable['max']-variable['min']+1))
                    index += 1

            elif variable['type'] == 'float':
                param.type = 'float'
                for dd in xrange(variable['size']):
                    val = variable['min'] + u[index]*(variable['max']-variable['min'])
                    val = variable['min'] if val < variable['min'] else val
                    val = variable['max'] if val > variable['max'] else val
                    param.dbl_val.append(val)
                    index += 1

            elif variable['type'] == 'enum':
                param.type = 'enum'
                for dd in xrange(variable['size']):
                    ii = self._index_map(u[index], len(variable['options']))
                    index += 1
                    param.str_val.append(variable['options'][ii])

            else:
                raise Exception("Unknown parameter type.")
            
            params.append(param)

        return params
            
    def card(self):
        return self.cardinality

    def _index_map(self, u, items):
        return int(np.floor((1-np.finfo(float).eps) * u * float(items)))

    def _index_unmap(self, u, items):
        return float(float(u) / float(items))
