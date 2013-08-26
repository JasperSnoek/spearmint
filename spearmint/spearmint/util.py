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
import re
import numpy        as np
import numpy.random as npr

def unpack_args(str):
    if len(str) > 1:
        eq_re = re.compile("\s*=\s*")
        return dict(map(lambda x: eq_re.split(x),
                        re.compile("\s*,\s*").split(str)))
    else:
        return {}

def slice_sample(init_x, logprob, sigma=1.0, step_out=True, max_steps_out=1000, 
                 compwise=False, verbose=False):
    def direction_slice(direction, init_x):
        def dir_logprob(z):
            return logprob(direction*z + init_x)
    
        upper = sigma*npr.rand()
        lower = upper - sigma
        llh_s = np.log(npr.rand()) + dir_logprob(0.0)
    
        l_steps_out = 0
        u_steps_out = 0
        if step_out:
            while dir_logprob(lower) > llh_s and l_steps_out < max_steps_out:
                l_steps_out += 1
                lower       -= sigma
            while dir_logprob(upper) > llh_s and u_steps_out < max_steps_out:
                u_steps_out += 1
                upper       += sigma
            
        steps_in = 0
        while True:
            steps_in += 1
            new_z     = (upper - lower)*npr.rand() + lower
            new_llh   = dir_logprob(new_z)
            if np.isnan(new_llh):
                print new_z, direction*new_z + init_x, new_llh, llh_s, init_x, logprob(init_x)
                raise Exception("Slice sampler got a NaN")
            if new_llh > llh_s:
                break
            elif new_z < 0:
                lower = new_z
            elif new_z > 0:
                upper = new_z
            else:
                raise Exception("Slice sampler shrank to zero!")

        if verbose:
            print "Steps Out:", l_steps_out, u_steps_out, " Steps In:", steps_in

        return new_z*direction + init_x
    
    if not init_x.shape:
        init_x = np.array([init_x])

    dims = init_x.shape[0]
    if compwise:
        ordering = range(dims)
        npr.shuffle(ordering)
        cur_x = init_x.copy()
        for d in ordering:
            direction    = np.zeros((dims))
            direction[d] = 1.0
            cur_x = direction_slice(direction, cur_x)
        return cur_x
            
    else:
        direction = npr.randn(dims)
        direction = direction / np.sqrt(np.sum(direction**2))
        return direction_slice(direction, init_x)
