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
    return GPEIperSecChooser(expt_dir, **args)

"""
Chooser module for the Gaussian process expected improvement per
second (EI) acquisition function.  Candidates are sampled densely in the unit
hypercube and then a subset of the most promising points are optimized to maximize
EI per second over hyperparameter samples.  Slice sampling is used to sample
Gaussian process hyperparameters for two GPs, one over the objective function and
the other over the running time of the algorithm.  
"""
class GPEIperSecChooser:

    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=10, 
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20):
        self.cov_func        = getattr(gp, covar)
        self.locker          = Locker()
        self.state_pkl       = os.path.join(expt_dir, self.__module__ + ".pkl")

        self.stats_file      = os.path.join(expt_dir, 
                                   self.__module__ + "_hyperparameters.txt")
        self.mcmc_iters      = int(mcmc_iters)
        self.burnin          = int(burnin)
        self.needs_burnin    = True
        self.pending_samples = pending_samples
        self.D               = -1
        self.hyper_iters     = 1
        # Number of points to optimize EI over
        self.grid_subset     = int(grid_subset)
        self.noiseless       = bool(int(noiseless))
        self.hyper_samples = []
        self.time_hyper_samples = []

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale  = 1    # zero-mean log normal prior
        self.max_ls      = 10    # top-hat prior on length scales

        self.time_noise_scale = 0.1  # horseshoe prior
        self.time_amp2_scale  = 1    # zero-mean log normal prior
        self.time_max_ls      = 10   # top-hat prior on length scales

    # A simple function to dump out hyperparameters to allow for a hot start
    # if the optimization is restarted.
    def dump_hypers(self):
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        self.locker.lock_wait(self.state_pkl)
        sys.stderr.write("...acquired\n")

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'dims'        : self.D,
                       'ls'          : self.ls,
                       'amp2'        : self.amp2,
                       'noise'       : self.noise,
                       'mean'        : self.mean,
                       'time_ls'     : self.time_ls,
                       'time_amp2'   : self.time_amp2,
                       'time_noise'  : self.time_noise,
                       'time_mean'   : self.time_mean },
                     fh)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.state_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.

        self.locker.unlock(self.state_pkl)

    def _real_init(self, dims, values, durations):
        
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        self.locker.lock_wait(self.state_pkl)
        sys.stderr.write("...acquired\n")

        if os.path.exists(self.state_pkl):          
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D          = state['dims']
            self.ls         = state['ls']
            self.amp2       = state['amp2']
            self.noise      = state['noise']
            self.mean       = state['mean']
            self.time_ls    = state['time_ls']
            self.time_amp2  = state['time_amp2']
            self.time_noise = state['time_noise']
            self.time_mean  = state['time_mean']
            self.needs_burnin = False
        else:

            # Input dimensionality.
            self.D = dims

            # Initial length scales.
            self.ls = np.ones(self.D)
            self.time_ls = np.ones(self.D)

            # Initial amplitude.
            self.amp2 = np.std(values)
            self.time_amp2 = np.std(durations)

            # Initial observation noise.
            self.noise = 1e-3
            self.time_noise = 1e-3

            # Initial mean.
            self.mean = np.mean(values)
            self.time_mean = np.mean(np.log(durations))

        self.locker.unlock(self.state_pkl)

    def cov(self, amp2, ls, x1, x2=None):
        if x2 is None:
            return amp2 * (self.cov_func(ls, x1, None) 
                           + 1e-6*np.eye(x1.shape[0]))
        else:
            return amp2 * self.cov_func(ls, x1, x2)

    # Given a set of completed 'experiments' in the unit hypercube with
    # corresponding objective 'values', pick from the next experiment to
    # run according to the acquisition function.
    def next(self, grid, values, durations,
             candidates, pending, complete):

        # Don't bother using fancy GP stuff at first.
        if complete.shape[0] < 2:
            return int(candidates[0])

        # Perform the real initialization.
        if self.D == -1:
            self._real_init(grid.shape[1], values[complete],
                            durations[complete])

        # Grab out the relevant sets.
        comp = grid[complete,:]
        cand = grid[candidates,:]
        pend = grid[pending,:]
        vals = values[complete]
        durs = durations[complete]

        # Bring time into the log domain before we do anything
        # to maintain strict positivity
        durs = np.log(durs)

        # Spray a set of candidates around the min so far 
        numcand = cand.shape[0]
        best_comp = np.argmin(vals)
        cand2 = np.vstack((np.random.randn(10,comp.shape[1])*0.001 +
                           comp[best_comp,:], cand))

        if self.mcmc_iters > 0:

            # Possibly burn in.
            if self.needs_burnin:
                for mcmc_iter in xrange(self.burnin):
                    self.sample_hypers(comp, vals, durs)
                    sys.stderr.write("BURN %d/%d] mean: %.2f  amp: %.2f "
                                     "noise: %.4f  min_ls: %.4f  max_ls: %.4f\n"
                                     % (mcmc_iter+1, self.burnin, self.mean,
                                        np.sqrt(self.amp2), self.noise,
                                        np.min(self.ls), np.max(self.ls)))
                self.needs_burnin = False
            
            # Sample from hyperparameters.
            # Adjust the candidates to hit ei/sec peaks
            self.hyper_samples = []
            for mcmc_iter in xrange(self.mcmc_iters):
                self.sample_hypers(comp, vals, durs)
                sys.stderr.write("%d/%d] mean: %.2f  amp: %.2f  noise: %.4f "
                                 "min_ls: %.4f  max_ls: %.4f\n"
                                 % (mcmc_iter+1, self.mcmc_iters, self.mean,
                                    np.sqrt(self.amp2), self.noise, 
                                    np.min(self.ls), np.max(self.ls)))

                sys.stderr.write("%d/%d] time_mean: %.2fs time_amp: %.2f  time_noise: %.4f "
                                 "time_min_ls: %.4f  time_max_ls: %.4f\n"
                                 % (mcmc_iter+1, self.mcmc_iters, np.exp(self.time_mean),
                                    np.sqrt(self.time_amp2), np.exp(self.time_noise), 
                                    np.min(self.time_ls), np.max(self.time_ls)))
            self.dump_hypers()

            # Pick the top candidates to optimize over                
            overall_ei = self.ei_over_hypers(comp,pend,cand2,vals,durs)
            inds = np.argsort(np.mean(overall_ei, axis=1))[-self.grid_subset:]
            cand2 = cand2[inds,:]

            # Adjust the candidates to hit ei peaks
            b = []# optimization bounds
            for i in xrange(0, cand.shape[1]):
                b.append((0, 1))
                
            for i in xrange(0, cand2.shape[0]):
                sys.stderr.write("Optimizing candidate %d/%d\n" %
                                 (i+1, cand2.shape[0]))
                #self.check_grad_ei_per(cand2[i,:].flatten(), comp, vals, durs)
                ret = spo.fmin_l_bfgs_b(self.grad_optimize_ei_over_hypers,
                                        cand2[i,:].flatten(),
                                        args=(comp,vals,durs,True),
                                        bounds=b, disp=0)
                cand2[i,:] = ret[0]

            cand = np.vstack((cand, cand2))

            overall_ei = self.ei_over_hypers(comp,pend,cand,vals,durs)
            best_cand = np.argmax(np.mean(overall_ei, axis=1))
            self.dump_hypers()
            if (best_cand >= numcand):
                return (int(numcand), cand[best_cand,:])

            return int(candidates[best_cand])

        else:
            # Optimize hyperparameters
            self.optimize_hypers(comp, vals, durs)

            sys.stderr.write("mean: %f  amp: %f  noise: %f "
                             "min_ls: %f  max_ls: %f\n"
                             % (self.mean, np.sqrt(self.amp2), 
                                self.noise, np.min(self.ls), np.max(self.ls)))

            # Pick the top candidates to optimize over
            ei = self.compute_ei_per_s(comp, pend, cand2, vals, durs)
            inds = np.argsort(np.mean(overall_ei, axis=1))[-self.grid_subset:]
            cand2 = cand2[inds,:]

            # Adjust the candidates to hit ei peaks
            b = []# optimization bounds
            for i in xrange(0, cand.shape[1]):
                b.append((0, 1))
                
            for i in xrange(0, cand2.shape[0]):
                sys.stderr.write("Optimizing candidate %d/%d\n" % 
                                 (i+1, cand2.shape[0]))
                ret = spo.fmin_l_bfgs_b(self.grad_optimize_ei,
                                        cand2[i,:].flatten(),
                                        args=(comp,vals,durs,True),
                                        bounds=b, disp=0)
                cand2[i,:] = ret[0]
                
            cand = np.vstack((cand, cand2))
            ei = self.compute_ei_per_s(comp, pend, cand, vals, durs)

            best_cand = np.argmax(ei)
            self.dump_hypers()

            if (best_cand >= numcand):
                return (int(numcand), cand[best_cand,:])

            return int(candidates[best_cand])

    # Compute EI over hyperparameter samples
    def ei_over_hypers(self,comp,pend,cand,vals,durs):
        overall_ei = np.zeros((cand.shape[0], self.mcmc_iters))
        for mcmc_iter in xrange(self.mcmc_iters):
            hyper = self.hyper_samples[mcmc_iter]
            time_hyper = self.time_hyper_samples[mcmc_iter]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]

            self.time_mean = time_hyper[0]
            self.time_noise = time_hyper[1]
            self.time_amp2 = time_hyper[2]
            self.time_ls = time_hyper[3]
            
            overall_ei[:,mcmc_iter] = self.compute_ei_per_s(comp, pend, cand,
                                                            vals, durs)
            
            return overall_ei

    def check_grad_ei_per(self, cand, comp, vals, durs):
        (ei,dx1) = self.grad_optimize_ei_over_hypers(cand, comp, vals, durs)
        dx2 = dx1*0
        idx = np.zeros(cand.shape[0])
        for i in xrange(0, cand.shape[0]):
            idx[i] = 1e-6*0.5
            (ei1,tmp) = self.grad_optimize_ei_over_hypers(cand + idx, comp, vals, durs)
            (ei2,tmp) = self.grad_optimize_ei_over_hypers(cand - idx, comp, vals, durs)
            dx2[i] = (ei - ei2)/(1e-6)
            idx[i] = 0
        print 'computed grads', dx1
        print 'finite diffs', dx2
        print (dx1/dx2)
        print np.sqrt(np.sum((dx1 - dx2)**2))
        time.sleep(2)

    # Adjust points by optimizing EI over a set of hyperparameter samples
    def grad_optimize_ei_over_hypers(self, cand, comp, vals, durs, compute_grad=True):
        summed_ei = 0
        summed_grad_ei = np.zeros(cand.shape).flatten()

        for mcmc_iter in xrange(self.mcmc_iters):
            hyper = self.hyper_samples[mcmc_iter]
            time_hyper = self.time_hyper_samples[mcmc_iter]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]

            self.time_mean = time_hyper[0]
            self.time_noise = time_hyper[1]
            self.time_amp2 = time_hyper[2]
            self.time_ls = time_hyper[3]

            if compute_grad:
                (ei,g_ei) = self.grad_optimize_ei(cand,comp,vals,durs,compute_grad)
                summed_grad_ei = summed_grad_ei + g_ei
            else:
                ei = self.grad_optimize_ei(cand,comp,vals,durs,compute_grad)
                
            summed_ei += ei

        if compute_grad:
            return (summed_ei, summed_grad_ei)
        else:
            return summed_ei

    def grad_optimize_ei(self, cand, comp, vals, durs, compute_grad=True):
        # Here we have to compute the gradients for ei per second
        # This means deriving through the two kernels, the one for predicting
        # time and the one predicting ei
        best = np.min(vals)
        cand = np.reshape(cand, (-1, comp.shape[1]))

        # First we make predictions for the durations
        # Compute covariances
        comp_time_cov   = self.cov(self.time_amp2, self.time_ls, comp)
        cand_time_cross = self.cov(self.time_amp2, self.time_ls,comp,cand)

        # Cholesky decompositions
        obsv_time_cov  = comp_time_cov + self.time_noise*np.eye(comp.shape[0])
        obsv_time_chol = spla.cholesky( obsv_time_cov, lower=True )

        # Linear systems
        t_alpha  = spla.cho_solve((obsv_time_chol, True), durs - self.time_mean)

        # Predict marginal mean times and (possibly) variances
        func_time_m = np.dot(cand_time_cross.T, t_alpha) + self.time_mean

        # We don't really need the time variances now
        #func_time_v = self.time_amp2*(1+1e-6) - np.sum(t_beta**2, axis=0)

        # Bring time out of the log domain
        func_time_m = np.exp(func_time_m)

        # Compute derivative of cross-distances.
        grad_cross_r = gp.grad_dist2(self.time_ls, comp, cand)

        # Apply covariance function
        cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
        cand_cross_grad = cov_grad_func(self.time_ls, comp, cand)
        grad_cross_t = np.squeeze(cand_cross_grad)        
            
        # Now compute the gradients w.r.t. ei
        # The primary covariances for prediction.
        comp_cov   = self.cov(self.amp2, self.ls, comp)
        cand_cross = self.cov(self.amp2, self.ls, comp, cand)
        
        # Compute the required Cholesky.
        obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
        obsv_chol = spla.cholesky( obsv_cov, lower=True )
        
        cand_cross_grad = cov_grad_func(self.ls, comp, cand)

        # Predictive things.                                   
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
        ei     = func_s*(u*ncdf + npdf)

        ei_per_s = -np.sum(ei/func_time_m)
        if not compute_grad:
            return ei

        grad_time_xp_m = np.dot(t_alpha.transpose(),grad_cross_t)

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5*npdf / func_s

        # Apply covariance function
        grad_cross = np.squeeze(cand_cross_grad)
        
        grad_xp_m = np.dot(alpha.transpose(),grad_cross)
        grad_xp_v = np.dot(-2*spla.cho_solve((obsv_chol, True),
                                             cand_cross).transpose(),grad_cross)
        
        grad_xp = self.amp2*(grad_xp_m*g_ei_m + grad_xp_v*g_ei_s2)
        grad_time_xp_m = self.time_amp2*grad_time_xp_m*func_time_m
        grad_xp = 0.5*(func_time_m*grad_xp - ei*grad_time_xp_m)/(func_time_m**2)

        return ei_per_s, grad_xp.flatten()

    def compute_ei_per_s(self, comp, pend, cand, vals, durs):
        # First we make predictions for the durations as that
        # doesn't depend on pending experiments

        # Compute covariances
        comp_time_cov   = self.cov(self.time_amp2, self.time_ls, comp)
        cand_time_cross = self.cov(self.time_amp2, self.time_ls,comp,cand)

        # Cholesky decompositions
        obsv_time_cov  = comp_time_cov + self.time_noise*np.eye(comp.shape[0])
        obsv_time_chol = spla.cholesky( obsv_time_cov, lower=True )

        # Linear systems
        t_alpha  = spla.cho_solve((obsv_time_chol, True), durs - self.time_mean)
        #t_beta   = spla.solve_triangular(obsv_time_chol, cand_time_cross, lower=True)

        # Predict marginal mean times and (possibly) variances
        func_time_m = np.dot(cand_time_cross.T, t_alpha) + self.time_mean
        # We don't really need the time variances now
        #func_time_v = self.time_amp2*(1+1e-6) - np.sum(t_beta**2, axis=0)

        # Bring time out of the log domain
        func_time_m = np.exp(func_time_m)
        
        if pend.shape[0] == 0:
            # If there are no pending, don't do anything fancy.

            # Current best.
            best = np.min(vals)

            # The primary covariances for prediction.
            comp_cov   = self.cov(self.amp2, self.ls, comp)      
            cand_cross = self.cov(self.amp2, self.ls, comp, cand)

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

            ei_per_s = ei/func_time_m
            return ei_per_s
        else:
            # If there are pending experiments, fantasize their outcomes.
            
            # Create a composite vector of complete and pending.
            comp_pend = np.concatenate((comp, pend))

            # Compute the covariance and Cholesky decomposition.
            comp_pend_cov  = self.cov(self.amp2, self.ls, comp_pend) + self.noise*np.eye(comp_pend.shape[0])
            comp_pend_chol = spla.cholesky(comp_pend_cov, lower=True)

            # Compute submatrices.
            pend_cross = self.cov(self.amp2, self.ls, comp, pend)
            pend_kappa = self.cov(self.amp2, self.ls, pend)

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
            pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0],self.pending_samples)) + self.mean

            # Include the fantasies.
            fant_vals = np.concatenate((np.tile(vals[:,np.newaxis], 
                                                (1,self.pending_samples)), pend_fant))

            # Compute bests over the fantasies.
            bests = np.min(fant_vals, axis=0)

            # Now generalize from these fantasies.
            cand_cross = self.cov(self.amp2, self.ls, comp_pend, cand)

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

            return np.divide(np.mean(ei, axis=1), func_time_m)

    def sample_hypers(self, comp, vals, durs):
        if self.noiseless:
            self.noise = 1e-3
            self._sample_noiseless(comp, vals)
        else:
            self._sample_noisy(comp, vals)
        self._sample_ls(comp, vals)

        self._sample_time_noisy(comp, durs)
        self._sample_time_ls(comp, durs)
        
        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))
        self.time_hyper_samples.append((self.time_mean, self.time_noise, self.time_amp2,
                                        self.time_ls))

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

    def _sample_time_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.time_max_ls):
                return -np.inf
            
            cov   = self.time_amp2 * (self.cov_func(ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + self.time_noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.time_mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-self.time_mean, solve)
            return lp

        self.time_ls = util.slice_sample(self.time_ls, logprob, compwise=True)

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
            
            cov   = amp2 * (self.cov_func(self.ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale/noise)**2))
            #lp -= 0.5*(np.log(noise)/self.noise_scale)**2

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.mean, self.amp2, self.noise]), logprob, compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = hypers[2]

    def _sample_time_noisy(self, comp, vals):
        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = hypers[2]

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0 or noise < 0:
                return -np.inf
            
            cov   = amp2 * (self.cov_func(self.time_ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.time_noise_scale/noise)**2))
            #lp -= 0.5*(np.log(noise)/self.time_noise_scale)**2

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.time_amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.time_mean, self.time_amp2, self.time_noise]), logprob, compwise=False)
        self.time_mean  = hypers[0]
        self.time_amp2  = hypers[1]
        self.time_noise = hypers[2]

    def _sample_noiseless(self, comp, vals):
        def logprob(hypers):
            mean  = hypers[0]
            amp2  = hypers[1]
            noise = 1e-3

            # This is pretty hacky, but keeps things sane.
            if mean > np.max(vals) or mean < np.min(vals):
                return -np.inf

            if amp2 < 0:
                return -np.inf
            
            cov   = amp2 * (self.cov_func(self.ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array([self.mean, self.amp2, self.noise]), logprob, compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = 1e-3

    def optimize_hypers(self, comp, vals, durs):
        # First the GP to observations
        mygp = gp.GP(self.cov_func.__name__)
        mygp.real_init(comp.shape[1], vals)
        mygp.optimize_hypers(comp,vals)
        self.mean = mygp.mean
        self.ls = mygp.ls
        self.amp2 = mygp.amp2
        self.noise = mygp.noise
        
        # Now the GP to times
        timegp = gp.GP(self.cov_func.__name__)
        timegp.real_init(comp.shape[1], durs)
        timegp.optimize_hypers(comp, durs)
        self.time_mean  = timegp.mean
        self.time_amp2  = timegp.amp2
        self.time_noise = timegp.noise
        self.time_ls    = timegp.ls

        # Save hyperparameter samples
        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))
        self.time_hyper_samples.append((self.time_mean, self.time_noise, self.time_amp2,
                                        self.time_ls))
        self.dump_hypers()
