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
import gp
import sys
import util
import tempfile
import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.stats    as sps
import scipy.optimize as spo
import cPickle

from Locker import *

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return GPEIChooser(expt_dir, **args)

"""
Chooser module for the Gaussian process expected improvement
acquisition function.  Candidates are sampled densely in the unit
hypercube and then the highest EI point is selected.  Slice sampling
is used to sample Gaussian process hyperparameters for the GP.
"""
class GPEIChooser:

    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10, 
                 pending_samples=100, noiseless=False):
        self.cov_func        = getattr(gp, covar)
        self.locker          = Locker()
        self.state_pkl       = os.path.join(expt_dir, self.__module__ + ".pkl")

        self.mcmc_iters      = int(mcmc_iters)
        self.pending_samples = pending_samples
        self.D               = -1
        self.hyper_iters     = 1
        self.noiseless       = bool(int(noiseless))

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale  = 1    # zero-mean log normal prior
        self.max_ls      = 2    # top-hat prior on length scales

    def __del__(self):
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        self.locker.lock_wait(self.state_pkl)
        sys.stderr.write("...acquired\n")

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'dims'   : self.D,
                       'ls'     : self.ls,
                       'amp2'   : self.amp2,
                       'noise'  : self.noise,
                       'mean'   : self.mean },
                     fh)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.state_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.

        self.locker.unlock(self.state_pkl)

    def _real_init(self, dims, values):
        
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        self.locker.lock_wait(self.state_pkl)
        sys.stderr.write("...acquired\n")

        if os.path.exists(self.state_pkl):            
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D     = state['dims']
            self.ls    = state['ls']
            self.amp2  = state['amp2']
            self.noise = state['noise']
            self.mean  = state['mean']
        else:

            # Input dimensionality.
            self.D = dims

            # Initial length scales.
            self.ls = np.ones(self.D)

            # Initial amplitude.
            self.amp2 = np.std(values)

            # Initial observation noise.
            self.noise = 1e-3

            # Initial mean.
            self.mean = np.mean(values)

        self.locker.unlock(self.state_pkl)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.amp2 * (self.cov_func(self.ls, x1, None) 
                               + 1e-6*np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)

    def next(self, grid, values, durations,
             candidates, pending, complete):

        # Don't bother using fancy GP stuff at first.
        if complete.shape[0] < 2:
            return int(candidates[0])

        # Perform the real initialization.
        if self.D == -1:
            self._real_init(grid.shape[1], values[complete])

        # Grab out the relevant sets.
        comp = grid[complete,:]
        cand = grid[candidates,:]
        pend = grid[pending,:]
        vals = values[complete]

        if self.mcmc_iters > 0:
            # Sample from hyperparameters.

            overall_ei = np.zeros((cand.shape[0], self.mcmc_iters))

            for mcmc_iter in xrange(self.mcmc_iters):

                self.sample_hypers(comp, vals)
                sys.stderr.write("mean: %f  amp: %f  noise: %f  min_ls: %f  max_ls: %f\n"
                                 % (self.mean, np.sqrt(self.amp2), self.noise, np.min(self.ls), np.max(self.ls)))

                overall_ei[:,mcmc_iter] = self.compute_ei(comp, pend, cand, vals)

            best_cand = np.argmax(np.mean(overall_ei, axis=1))

            return int(candidates[best_cand])

        else:
            # Optimize hyperparameters
            try:
                self.optimize_hypers(comp, vals)
            except:
                # Initial length scales.
                self.ls = np.ones(self.D)
                # Initial amplitude.
                self.amp2 = np.std(vals)
                # Initial observation noise.
                self.noise = 1e-3
            sys.stderr.write("mean: %f  amp: %f  noise: %f  min_ls: %f  max_ls: %f\n"
                             % (self.mean, np.sqrt(self.amp2), self.noise, np.min(self.ls),
                                np.max(self.ls)))

            ei = self.compute_ei(comp, pend, cand, vals)

            best_cand = np.argmax(ei)

            return int(candidates[best_cand])

    def compute_ei(self, comp, pend, cand, vals):
        if pend.shape[0] == 0:
            # If there are no pending, don't do anything fancy.

            # Current best.
            best = np.min(vals)

            # The primary covariances for prediction.
            comp_cov   = self.cov(comp)      
            cand_cross = self.cov(comp, cand)

            # Compute the required Cholesky.
            obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
            obsv_chol = spla.cholesky( obsv_cov, lower=True )

            # Solve the linear systems.
            alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
            beta   = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

            # Predict the marginal means and variances at candidates.
            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)

            # Expected improvement
            func_s = np.sqrt(func_v)
            u      = (best - func_m) / func_s
            ncdf   = sps.norm.cdf(u)
            npdf   = sps.norm.pdf(u)
            ei     = func_s*( u*ncdf + npdf)

            return ei
        else:
            # If there are pending experiments, fantasize their outcomes.
            
            # Create a composite vector of complete and pending.
            comp_pend = np.concatenate((comp, pend))

            # Compute the covariance and Cholesky decomposition.
            comp_pend_cov  = self.cov(comp_pend) + self.noise*np.eye(comp_pend.shape[0])
            comp_pend_chol = spla.cholesky(comp_pend_cov, lower=True)

            # Compute submatrices.
            pend_cross = self.cov(comp, pend)
            pend_kappa = self.cov(pend)

            # Use the sub-Cholesky.
            obsv_chol = comp_pend_chol[:comp.shape[0],:comp.shape[0]]

            # Solve the linear systems.
            alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
            beta   = spla.cho_solve((obsv_chol, True), pend_cross)

            # Finding predictive means and variances.
            pend_m = np.dot(pend_cross.T, alpha) + self.mean
            pend_K = pend_kappa - np.dot(pend_cross.T, beta)

            # Take the Cholesky of the predictive covariance.
            pend_chol = spla.cholesky(pend_K, lower=True)

            # Make predictions.
            pend_fant = (np.dot(pend_chol, npr.randn(pend.shape[0],self.pending_samples))
                         + self.mean)

            # Include the fantasies.
            fant_vals = np.concatenate((np.tile(vals[:,np.newaxis], 
                                                (1,self.pending_samples)), pend_fant))

            # Compute bests over the fantasies.
            bests = np.min(fant_vals, axis=0)

            # Now generalize from these fantasies.
            cand_cross = self.cov(comp_pend, cand)

            # Solve the linear systems.
            alpha  = spla.cho_solve((comp_pend_chol, True), fant_vals - self.mean)
            beta   = spla.solve_triangular(comp_pend_chol, cand_cross, lower=True)

            # Predict the marginal means and variances at candidates.
            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)

            # Expected improvement
            func_s = np.sqrt(func_v[:,np.newaxis])
            u      = (bests[np.newaxis,:] - func_m) / func_s
            ncdf   = sps.norm.cdf(u)
            npdf   = sps.norm.pdf(u)
            ei     = func_s*( u*ncdf + npdf)

            return np.mean(ei, axis=1)

    def sample_hypers(self, comp, vals):
        if self.noiseless:
            self.noise = 1e-3
            self._sample_noiseless(comp, vals)
        else:
            self._sample_noisy(comp, vals)
        self._sample_ls(comp, vals)

    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf
            
            cov   = self.amp2 * (self.cov_func(ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + self.noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-self.mean, solve)
            return lp

        self.ls = util.slice_sample(self.ls, logprob, compwise=True)

    def _sample_noisy(self, comp, vals):
        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = hypers[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf
            
            cov   = amp2 * (self.cov_func(self.ls, comp, None) +
                            1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale/noise)**2))

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.mean, self.amp2, self.noise]),
                                   logprob, compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = hypers[2]

    def _sample_noiseless(self, comp, vals):
        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = 1e-3

            if amp2 < 0:
                return -np.inf
            
            cov   = amp2 * (self.cov_func(self.ls, comp, None) +
                            1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.mean, self.amp2, self.noise]), logprob,
                                   compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = 1e-3

    def optimize_hypers(self, comp, vals):
        mygp = gp.GP(self.cov_func.__name__)
        mygp.real_init(comp.shape[1], vals)
        mygp.optimize_hypers(comp,vals)
        self.mean = mygp.mean
        self.ls = mygp.ls
        self.amp2 = mygp.amp2
        self.noise = mygp.noise
        
        # Save hyperparameter samples
        #self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))
        #self.dump_hypers()
        
        return

