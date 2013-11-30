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
from spearmint import gp
import sys
from spearmint import util
import tempfile
import copy
import numpy          as np
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.stats    as sps
import scipy.optimize as spo
import cPickle
import multiprocessing

from helpers import *
from Locker  import *

def optimize_pt(c, b, comp, pend, vals, model):
    ret = spo.fmin_l_bfgs_b(model.grad_optimize_ei_over_hypers,
                            c.flatten(), args=(comp, pend, vals),
                            bounds=b, disp=0)
    return ret[0]

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return GPEIOptChooser(expt_dir, **args)
"""
Chooser module for the Gaussian process expected improvement (EI)
acquisition function where points are sampled densely in the unit
hypercube and then a subset of the points are optimized to maximize EI
over hyperparameter samples.  Slice sampling is used to sample
Gaussian process hyperparameters.
"""
class GPEIOptChooser:

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
        self.pending_samples = int(pending_samples)
        self.D               = -1
        self.hyper_iters     = 1
        # Number of points to optimize EI over
        self.grid_subset     = int(grid_subset)
        self.noiseless       = bool(int(noiseless))
        self.hyper_samples = []

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale  = 1    # zero-mean log normal prior
        self.max_ls      = 2    # top-hat prior on length scales

    def dump_hypers(self):
        self.locker.lock_wait(self.state_pkl)

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({ 'dims'          : self.D,
                       'ls'            : self.ls,
                       'amp2'          : self.amp2,
                       'noise'         : self.noise,
                       'hyper_samples' : self.hyper_samples,
                       'mean'          : self.mean },
                     fh)
        fh.close()

        # Use an atomic move for better NFS happiness.
        cmd = 'mv "%s" "%s"' % (fh.name, self.state_pkl)
        os.system(cmd) # TODO: Should check system-dependent return status.

        self.locker.unlock(self.state_pkl)

        # Write the hyperparameters out to a human readable file as well
        fh    = open(self.stats_file, 'w')
        fh.write('Mean Noise Amplitude <length scales>\n')
        fh.write('-----------ALL SAMPLES-------------\n')
        meanhyps = 0*np.hstack(self.hyper_samples[0])
        for i in self.hyper_samples:
            hyps = np.hstack(i)
            meanhyps += (1/float(len(self.hyper_samples)))*hyps
            for j in hyps:
                fh.write(str(j) + ' ')
            fh.write('\n')

        fh.write('-----------MEAN OF SAMPLES-------------\n')
        for j in meanhyps:
            fh.write(str(j) + ' ')
        fh.write('\n')
        fh.close()

    # This passes out html or javascript to display interesting
    # stats - such as the length scales (sensitivity to various
    # dimensions).
    def generate_stats_html(self):
        # Need this because the model may not necessarily be
        # initialized when this code is called.
        if not self._read_only():
            return 'Chooser not yet ready to display output'

        mean_mean  = np.mean(np.vstack([h[0] for h in self.hyper_samples]))
        mean_noise = np.mean(np.vstack([h[1] for h in self.hyper_samples]))
        mean_ls    = np.mean(np.vstack([h[3][np.newaxis,:] for h in self.hyper_samples]),0)

        try:
            output = (
                '<br /><span class=\"label label-info\">Estimated mean:</span> ' + str(mean_mean) + 
                '<br /><span class=\"label label-info\">Estimated noise:</span> ' + str(mean_noise) + 
                '<br /><br /><span class=\"label label-info\">Inverse parameter sensitivity' +
                ' - Gaussian Process length scales</span><br /><br />' +
                '<div id=\"lschart\"></div><script type=\"text/javascript\">' +
                'var lsdata = [' + ','.join(['%.2f' % i for i in mean_ls]) + '];')
        except:
            return 'Chooser not yet ready to display output.'

        output += ('bar_chart("#lschart", lsdata, ' + str(self.max_ls) + ');' +
                   '</script>')
        return output

    # Read in the chooser from file. Returns True only on success
    def _read_only(self):
        if os.path.exists(self.state_pkl):
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D             = state['dims']
            self.ls            = state['ls']
            self.amp2          = state['amp2']
            self.noise         = state['noise']
            self.mean          = state['mean']
            self.hyper_samples = state['hyper_samples']
            self.needs_burnin  = False
            return True

        return False

    def _real_init(self, dims, values):
        self.locker.lock_wait(self.state_pkl)

        self.randomstate = npr.get_state()
        if os.path.exists(self.state_pkl):
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D             = state['dims']
            self.ls            = state['ls']
            self.amp2          = state['amp2']
            self.noise         = state['noise']
            self.mean          = state['mean']
            self.hyper_samples = state['hyper_samples']
            self.needs_burnin  = False
        else:

            # Input dimensionality.
            self.D = dims

            # Initial length scales.
            self.ls = np.ones(self.D)

            # Initial amplitude.
            self.amp2 = np.std(values)+1e-4

            # Initial observation noise.
            self.noise = 1e-3

            # Initial mean.
            self.mean = np.mean(values)

            # Save hyperparameter samples
            self.hyper_samples.append((self.mean, self.noise, self.amp2,
                                       self.ls))

        self.locker.unlock(self.state_pkl)

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.amp2 * (self.cov_func(self.ls, x1, None)
                               + 1e-6*np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)

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
            self._real_init(grid.shape[1], values[complete])

        # Grab out the relevant sets.
        comp = grid[complete,:]
        cand = grid[candidates,:]
        pend = grid[pending,:]
        vals = values[complete]
        numcand = cand.shape[0]

        # Spray a set of candidates around the min so far
        best_comp = np.argmin(vals)
        cand2 = np.vstack((np.random.randn(10,comp.shape[1])*0.001 +
                           comp[best_comp,:], cand))

        if self.mcmc_iters > 0:

            # Possibly burn in.
            if self.needs_burnin:
                for mcmc_iter in xrange(self.burnin):
                    self.sample_hypers(comp, vals)
                    log("BURN %d/%d] mean: %.2f  amp: %.2f "
                                     "noise: %.4f  min_ls: %.4f  max_ls: %.4f"
                                     % (mcmc_iter+1, self.burnin, self.mean,
                                        np.sqrt(self.amp2), self.noise,
                                        np.min(self.ls), np.max(self.ls)))
                self.needs_burnin = False

            # Sample from hyperparameters.
            # Adjust the candidates to hit ei peaks
            self.hyper_samples = []
            for mcmc_iter in xrange(self.mcmc_iters):
                self.sample_hypers(comp, vals)
                log("%d/%d] mean: %.2f  amp: %.2f  noise: %.4f "
                                 "min_ls: %.4f  max_ls: %.4f"
                                 % (mcmc_iter+1, self.mcmc_iters, self.mean,
                                    np.sqrt(self.amp2), self.noise,
                                    np.min(self.ls), np.max(self.ls)))
            self.dump_hypers()

            b = []# optimization bounds
            for i in xrange(0, cand.shape[1]):
                b.append((0, 1))

            overall_ei = self.ei_over_hypers(comp,pend,cand2,vals)
            inds = np.argsort(np.mean(overall_ei,axis=1))[-self.grid_subset:]
            cand2 = cand2[inds,:]

            # This is old code to optimize each point in parallel. Uncomment
            # and replace if multiprocessing doesn't work
            #for i in xrange(0, cand2.shape[0]):
            #    log("Optimizing candidate %d/%d" %
            #                     (i+1, cand2.shape[0]))
            #self.check_grad_ei(cand2[i,:].flatten(), comp, pend, vals)
            #    ret = spo.fmin_l_bfgs_b(self.grad_optimize_ei_over_hypers,
            #                            cand2[i,:].flatten(), args=(comp,pend,vals),
            #                            bounds=b, disp=0)
            #    cand2[i,:] = ret[0]
            #cand = np.vstack((cand, cand2))

            # Optimize each point in parallel
            pool = multiprocessing.Pool(self.grid_subset)
            results = [pool.apply_async(optimize_pt,args=(
                        c,b,comp,pend,vals,copy.copy(self))) for c in cand2]
            for res in results:
                cand = np.vstack((cand, res.get(1e8)))
            pool.close()

            overall_ei = self.ei_over_hypers(comp,pend,cand,vals)
            best_cand = np.argmax(np.mean(overall_ei, axis=1))

            if (best_cand >= numcand):
                return (int(numcand), cand[best_cand,:])

            return int(candidates[best_cand])

        else:
            # Optimize hyperparameters
            self.optimize_hypers(comp, vals)

            log("mean: %.2f  amp: %.2f  noise: %.4f  "
                             "min_ls: %.4f  max_ls: %.4f"
                             % (self.mean, np.sqrt(self.amp2), self.noise,
                                np.min(self.ls), np.max(self.ls)))

            # Optimize over EI
            b = []# optimization bounds
            for i in xrange(0, cand.shape[1]):
                b.append((0, 1))

            for i in xrange(0, cand2.shape[0]):
                ret = spo.fmin_l_bfgs_b(self.grad_optimize_ei,
                                        cand2[i,:].flatten(), args=(comp,vals,True),
                                        bounds=b, disp=0)
                cand2[i,:] = ret[0]
            cand = np.vstack((cand, cand2))

            ei = self.compute_ei(comp, pend, cand, vals)
            best_cand = np.argmax(ei)

            if (best_cand >= numcand):
                return (int(numcand), cand[best_cand,:])

            return int(candidates[best_cand])

    # Compute EI over hyperparameter samples
    def ei_over_hypers(self,comp,pend,cand,vals):
        overall_ei = np.zeros((cand.shape[0], self.mcmc_iters))
        for mcmc_iter in xrange(self.mcmc_iters):
            hyper = self.hyper_samples[mcmc_iter]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            overall_ei[:,mcmc_iter] = self.compute_ei(comp, pend, cand,
                                                      vals)
        return overall_ei

    def check_grad_ei(self, cand, comp, pend, vals):
        (ei,dx1) = self.grad_optimize_ei_over_hypers(cand, comp, pend, vals)
        dx2 = dx1*0
        idx = np.zeros(cand.shape[0])
        for i in xrange(0, cand.shape[0]):
            idx[i] = 1e-6
            (ei1,tmp) = self.grad_optimize_ei_over_hypers(cand + idx, comp, pend, vals)
            (ei2,tmp) = self.grad_optimize_ei_over_hypers(cand - idx, comp, pend, vals)
            dx2[i] = (ei - ei2)/(2*1e-6)
            idx[i] = 0
        print 'computed grads', dx1
        print 'finite diffs', dx2
        print (dx1/dx2)
        print np.sum((dx1 - dx2)**2)
        time.sleep(2)

    # Adjust points by optimizing EI over a set of hyperparameter samples
    def grad_optimize_ei_over_hypers(self, cand, comp, pend, vals, compute_grad=True):
        summed_ei = 0
        summed_grad_ei = np.zeros(cand.shape).flatten()
        ls = self.ls.copy()
        amp2 = self.amp2
        mean = self.mean
        noise = self.noise

        for hyper in self.hyper_samples:
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]
            if compute_grad:
                (ei,g_ei) = self.grad_optimize_ei(cand,comp,pend,vals,compute_grad)
                summed_grad_ei = summed_grad_ei + g_ei
            else:
                ei = self.grad_optimize_ei(cand,comp,pend,vals,compute_grad)
            summed_ei += ei

        self.mean = mean
        self.amp2 = amp2
        self.noise = noise
        self.ls = ls.copy()

        if compute_grad:
            return (summed_ei, summed_grad_ei)
        else:
            return summed_ei

    # Adjust points based on optimizing their ei
    def grad_optimize_ei(self, cand, comp, pend, vals, compute_grad=True):
        if pend.shape[0] == 0:
            best = np.min(vals)
            cand = np.reshape(cand, (-1, comp.shape[1]))

            # The primary covariances for prediction.
            comp_cov   = self.cov(comp)
            cand_cross = self.cov(comp, cand)

            # Compute the required Cholesky.
            obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
            obsv_chol = spla.cholesky(obsv_cov, lower=True)

            cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
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
            ei     = func_s*( u*ncdf + npdf)

            if not compute_grad:
                return ei

            # Gradients of ei w.r.t. mean and variance
            g_ei_m = -ncdf
            g_ei_s2 = 0.5*npdf / func_s

            # Apply covariance function
            grad_cross = np.squeeze(cand_cross_grad)

            grad_xp_m = np.dot(alpha.transpose(),grad_cross)
            grad_xp_v = np.dot(-2*spla.cho_solve(
                    (obsv_chol, True),cand_cross).transpose(), grad_cross)

            grad_xp = 0.5*self.amp2*(grad_xp_m*g_ei_m + grad_xp_v*g_ei_s2)
            ei = -np.sum(ei)

            return ei, grad_xp.flatten()

        else:
            # If there are pending experiments, fantasize their outcomes.
            cand = np.reshape(cand, (-1, comp.shape[1]))

            # Create a composite vector of complete and pending.
            comp_pend = np.concatenate((comp, pend))

            # Compute the covariance and Cholesky decomposition.
            comp_pend_cov  = (self.cov(comp_pend) +
                              self.noise*np.eye(comp_pend.shape[0]))
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
            npr.set_state(self.randomstate)
            pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0],self.pending_samples)) + pend_m[:,None]

            # Include the fantasies.
            fant_vals = np.concatenate(
                (np.tile(vals[:,np.newaxis],
                         (1,self.pending_samples)), pend_fant))

            # Compute bests over the fantasies.
            bests = np.min(fant_vals, axis=0)

            # Now generalize from these fantasies.
            cand_cross = self.cov(comp_pend, cand)
            cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
            cand_cross_grad = cov_grad_func(self.ls, comp_pend, cand)

            # Solve the linear systems.
            alpha  = spla.cho_solve((comp_pend_chol, True),
                                    fant_vals - self.mean)
            beta   = spla.solve_triangular(comp_pend_chol, cand_cross,
                                           lower=True)

            # Predict the marginal means and variances at candidates.
            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)

            # Expected improvement
            func_s = np.sqrt(func_v[:,np.newaxis])
            u      = (bests[np.newaxis,:] - func_m) / func_s
            ncdf   = sps.norm.cdf(u)
            npdf   = sps.norm.pdf(u)
            ei     = func_s*( u*ncdf + npdf)

            # Gradients of ei w.r.t. mean and variance
            g_ei_m = -ncdf
            g_ei_s2 = 0.5*npdf / func_s

            # Apply covariance function
            grad_cross = np.squeeze(cand_cross_grad)

            grad_xp_m = np.dot(alpha.transpose(),grad_cross)
            grad_xp_v = np.dot(-2*spla.cho_solve(
                    (comp_pend_chol, True),cand_cross).transpose(), grad_cross)

            grad_xp = 0.5*self.amp2*(grad_xp_m*np.tile(g_ei_m,(comp.shape[1],1)).T + (grad_xp_v.T*g_ei_s2).T)
            ei = -np.mean(ei, axis=1)
            grad_xp = np.mean(grad_xp,axis=0)

            return ei, grad_xp.flatten()

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
            comp_pend_cov  = (self.cov(comp_pend) +
                              self.noise*np.eye(comp_pend.shape[0]))
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
            npr.set_state(self.randomstate)
            pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0],self.pending_samples)) + pend_m[:,None]

            # Include the fantasies.
            fant_vals = np.concatenate(
                (np.tile(vals[:,np.newaxis],
                         (1,self.pending_samples)), pend_fant))

            # Compute bests over the fantasies.
            bests = np.min(fant_vals, axis=0)

            # Now generalize from these fantasies.
            cand_cross = self.cov(comp_pend, cand)

            # Solve the linear systems.
            alpha  = spla.cho_solve((comp_pend_chol, True),
                                    fant_vals - self.mean)
            beta   = spla.solve_triangular(comp_pend_chol, cand_cross,
                                           lower=True)

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
        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))

    def _sample_ls(self, comp, vals):
        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.max_ls):
                return -np.inf

            cov   = (self.amp2 * (self.cov_func(ls, comp, None) +
                1e-6*np.eye(comp.shape[0])) + self.noise*np.eye(comp.shape[0]))
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp    = (-np.sum(np.log(np.diag(chol))) -
                      0.5*np.dot(vals-self.mean, solve))
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

            cov   = (amp2 * (self.cov_func(self.ls, comp, None) +
                1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0]))
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale/noise)**2))

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array(
                [self.mean, self.amp2, self.noise]), logprob, compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = hypers[2]

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

            cov   = (amp2 * (self.cov_func(self.ls, comp, None) +
                1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0]))
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array(
                [self.mean, self.amp2, self.noise]), logprob, compwise=False)
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
        self.hyper_samples.append((self.mean, self.noise, self.amp2, self.ls))
        self.dump_hypers()

        return
