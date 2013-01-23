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
import math
import numpy.random   as npr
import scipy.linalg   as spla
import scipy.stats    as sps
import scipy.optimize as spo
import cPickle
import matplotlib.pyplot as plt
import multiprocessing
import copy

from Locker import *

# Wrapper function to pass to parallel ei optimization calls
def optimize_pt(c, b, comp, pend, vals, labels, model):
    ret = spo.fmin_l_bfgs_b(model.grad_optimize_ei_over_hypers,
                            c.flatten(), args=(comp, pend, vals, labels),
                            bounds=b, disp=0)
    return ret[0]

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return GPConstrainedEIChooser(expt_dir, **args)

"""
Chooser module for Constrained Gaussian process expected improvement.
Candidates are sampled densely in the unit hypercube and then a subset
of the most promising points are optimized to maximize constrained EI 
over hyperparameter samples.  Slice sampling is used to sample
Gaussian process hyperparameters for two GPs, one over the objective
function and the other a probit likelihood classification GP that estimates the
probability that a point is outside of the constraint space.
"""
class GPConstrainedEIChooser:

    def __init__(self, expt_dir, covar="Matern52", mcmc_iters=20, 
                 pending_samples=100, noiseless=False, burnin=100,
                 grid_subset=20, constraint_violating_value=np.inf,
                 verbosity=0, visualize2D=False):
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
        self.hyper_samples   = []
        self.constraint_hyper_samples = []
        self.ff              = None
        self.ff_samples      = []
        self.verbosity       = int(verbosity)

        self.noise_scale = 0.1  # horseshoe prior
        self.amp2_scale  = 1    # zero-mean log normal prior
        self.max_ls      = 2    # top-hat prior on length scales

        self.constraint_noise_scale = 0.1  # horseshoe prior
        self.constraint_amp2_scale  = 1    # zero-mean log normal prio
        self.constraint_gain        = 1   # top-hat prior on length scales
        self.constraint_max_ls      = 2   # top-hat prior on length scales
        self.bad_value = float(constraint_violating_value)
        self.visualize2D            = visualize2D

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
                       'constraint_ls'     : self.constraint_ls,
                       'constraint_amp2'   : self.constraint_amp2,
                       'constraint_noise'  : self.constraint_noise,
                       'constraint_mean'   : self.constraint_mean },
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

    def _real_init(self, dims, values, durations):
        
        sys.stderr.write("Waiting to lock hyperparameter pickle...")
        self.locker.lock_wait(self.state_pkl)
        sys.stderr.write("...acquired\n")

        self.randomstate = npr.get_state()
        if os.path.exists(self.state_pkl):            
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            self.D                = state['dims']
            self.ls               = state['ls']
            self.amp2             = state['amp2']
            self.noise            = state['noise']
            self.mean             = state['mean']
            self.constraint_ls    = state['constraint_ls']
            self.constraint_amp2  = state['constraint_amp2']
            self.constraint_noise = state['constraint_noise']
            self.constraint_mean  = state['constraint_mean']
            self.constraint_gain  = state['constraint_mean']
            self.needs_burnin     = False
        else:

            # Identify constraint violations
            # Note that we'll treat NaNs and Infs as these values as well 
            # as an optional user defined value
            goodvals = np.nonzero(np.logical_and(values != self.bad_value,
                                                 np.isfinite(values)))[0]

            # Input dimensionality.
            self.D = dims

            # Initial length scales.
            self.ls = np.ones(self.D)
            self.constraint_ls = np.ones(self.D)

            # Initial amplitude.
            self.amp2 = np.std(values[goodvals])
            self.constraint_amp2 = 1#np.std(durations)

            # Initial observation noise.
            self.noise = 1e-3
            self.constraint_noise = 1e-3
            self.constraint_gain = 1

            # Initial mean.
            self.mean = np.mean(values[goodvals])
            self.constraint_mean = 0.5

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

        # Grab out the relevant sets.
        comp = grid[complete,:]
        cand = grid[candidates,:]
        pend = grid[pending,:]
        vals = values[complete]

        # Identify constraint violations
        # Note that we'll treat NaNs and Infs as these values as well 
        # as an optional user defined value
        idx = np.logical_and(vals != self.bad_value,
                             np.isfinite(vals))
        goodvals = np.nonzero(idx)[0]
        badvals = np.nonzero(np.logical_not(idx))[0]

        print 'Found %d constraint violating jobs' % (badvals.shape[0])

        # There's no point regressing on one observation
        print 'Received %d valid results' % (goodvals.shape[0])
        if goodvals.shape[0] < 2:
            return int(candidates[0])

        labels = np.zeros(vals.shape[0])
        labels[goodvals] = 1

        if np.sum(labels) < 2:
            return int(candidates[0])

        # Perform the real initialization.
        if self.D == -1:
            self._real_init(grid.shape[1], values[complete],
                            durations[complete])

        # Spray a set of candidates around the min so far 
        numcand = cand.shape[0]
        best_comp = np.argmin(vals)
        cand2 = np.vstack((np.random.randn(10,comp.shape[1])*0.001 +
                           comp[best_comp,:], cand))

        if self.mcmc_iters > 0:

            # Possibly burn in.
            if self.needs_burnin:
                for mcmc_iter in xrange(self.burnin):
                    self.sample_constraint_hypers(comp, labels)
                    self.sample_hypers(comp[goodvals,:], vals[goodvals])
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
                self.sample_constraint_hypers(comp, labels)
                self.sample_hypers(comp[goodvals,:], vals[goodvals])
                if self.verbosity > 0:
                    sys.stderr.write("%d/%d] mean: %.2f  amp: %.2f noise: %.4f "
                                     "min_ls: %.4f  max_ls: %.4f\n"
                                     % (mcmc_iter+1, self.mcmc_iters, self.mean,
                                        np.sqrt(self.amp2), self.noise,
                                        np.min(self.ls), np.max(self.ls)))

                    sys.stderr.write("%d/%d] constraint_mean: %.2f "
                                     "constraint_amp: %.2f "
                                     "constraint_gain: %.4f "
                                     "constraint_min_ls: %.4f "
                                     "constraint_max_ls: "
                                     "%.4f\n"
                                     % (mcmc_iter+1, self.mcmc_iters, 
                                        self.constraint_mean,
                                        np.sqrt(self.constraint_amp2), 
                                        self.constraint_gain,
                                        np.min(self.constraint_ls), 
                                        np.max(self.constraint_ls)))
            self.dump_hypers()
            comp_preds = np.zeros(labels.shape[0]).flatten()
            
            preds = self.pred_constraint_voilation(cand, comp, labels).flatten()
            for ii in xrange(self.mcmc_iters):
                constraint_hyper = self.constraint_hyper_samples[ii]            
                self.ff = self.ff_samples[ii]
                self.constraint_mean = constraint_hyper[0]
                self.constraint_gain = constraint_hyper[1]
                self.constraint_amp2 = constraint_hyper[2]
                self.constraint_ls = constraint_hyper[3]
                comp_preds += self.pred_constraint_voilation(comp, comp, 
                                                             labels).flatten()
            comp_preds = comp_preds / float(self.mcmc_iters)
            print 'Predicting %.2f%% constraint violations (%d/%d): ' % (
            np.mean(preds < 0.5)*100, np.sum(preds < 0.5), preds.shape[0])
            if self.verbosity > 0:
                print 'Prediction` %f%% train accuracy (%d/%d): ' % (
                    np.mean((comp_preds > 0.5) == labels), 
                    np.sum((comp_preds > 0.5) == labels), comp_preds.shape[0])

            if self.visualize2D:
                delta = 0.025
                x = np.arange(0, 1.0, delta)
                y = np.arange(0, 1.0, delta)
                X, Y = np.meshgrid(x, y)

                cpreds = np.zeros((X.shape[0], X.shape[1]))
                predei = np.zeros((X.shape[0], X.shape[1]))
                predei2 = np.zeros((X.shape[0], X.shape[1]))
                for ii in xrange(self.mcmc_iters):
                    constraint_hyper = self.constraint_hyper_samples[ii]
                    self.ff = self.ff_samples[ii]
                    self.constraint_mean = constraint_hyper[0]
                    self.constraint_gain = constraint_hyper[1]
                    self.constraint_amp2 = constraint_hyper[2]
                    self.constraint_ls = constraint_hyper[3]

                    cpred = self.pred_constraint_voilation(
                        np.hstack((X.flatten()[:,np.newaxis], 
                                   Y.flatten()[:,np.newaxis])), comp, labels)
                    pei = self.compute_constrained_ei(
                        comp, pend, np.hstack(
                            (X.flatten()[:,np.newaxis], 
                             Y.flatten()[:,np.newaxis])), vals, labels)
                    pei2 = self.compute_ei(
                        comp, pend, np.hstack(
                            (X.flatten()[:,np.newaxis], 
                             Y.flatten()[:,np.newaxis])), vals, labels)
                
                    cpreds += np.reshape(cpred, (X.shape[0], X.shape[1]))
                    predei += np.reshape(pei, (X.shape[0], X.shape[1]))
                    predei2 += np.reshape(pei2, (X.shape[0], X.shape[1]))

                plt.figure(1)
                plt.clf()
                cpreds = cpreds/float(self.mcmc_iters)
                CS = plt.contour(X,Y,cpreds)
                plt.clabel(CS, inline=1, fontsize=10)
                plt.plot(comp[labels == 0,0], comp[labels == 0,1], 'rx')
                plt.plot(comp[labels == 1,0], comp[labels == 1,1], 'bx')
                plt.title(
                    'Contours of Classification GP (Prob of not being a '
                    'constraint violation)')
                plt.legend(('Constraint Violations', 'Good points'),
                           'lower left')
                plt.savefig('constrained_ei_chooser_class_contour.pdf')

                plt.figure(2)
                plt.clf()
                predei = predei/float(self.mcmc_iters)
                CS = plt.contour(X,Y,predei)
                plt.clabel(CS, inline=1, fontsize=10)
                plt.plot(comp[labels == 0,0], comp[labels == 0,1], 'rx')
                plt.plot(comp[labels == 1,0], comp[labels == 1,1], 'bx')
                plt.title('Contours of EI*P(not violating constraint)')
                plt.legend(('Constraint Violations', 'Good points'),
                           'lower left')
                plt.savefig('constrained_ei_chooser_eitimesprob_contour.pdf')

                plt.figure(3)
                plt.clf()
                predei2 = predei2/float(self.mcmc_iters)
                CS = plt.contour(X,Y,predei2)
                plt.clabel(CS, inline=1, fontsize=10)
                plt.plot(comp[labels == 0,0], comp[labels == 0,1], 'rx')
                plt.plot(comp[labels == 1,0], comp[labels == 1,1], 'bx')
                plt.title('Contours of EI')
                plt.legend(('Constraint Violations', 'Good points'),
                           'lower left')
                plt.savefig('constrained_ei_chooser_ei_contour.pdf')
                #plt.show()

            # Pick the top candidates to optimize over                
            overall_ei = self.ei_over_hypers(comp,pend,cand2,vals,labels)
            inds = np.argsort(np.mean(overall_ei, axis=1))[-self.grid_subset:]
            cand2 = cand2[inds,:]

            # Adjust the candidates to hit ei peaks
            b = []# optimization bounds
            for i in xrange(0, cand.shape[1]):
                b.append((0, 1))
                
            # Optimize each point in parallel
            pool = multiprocessing.Pool(self.grid_subset)
            results = [pool.apply_async(optimize_pt,args=(
                        c,b,comp,pend,vals,labels, copy.copy(self))) for c in cand2]            
            for res in results:
                cand = np.vstack((cand, res.get(1024)))
            pool.close()

            #for i in xrange(0, cand2.shape[0]):
            #    sys.stderr.write("Optimizing candidate %d/%d\n" %
            #                     (i+1, cand2.shape[0]))
            #    self.check_grad_ei(cand2[i,:], comp, pend, vals, labels)
            #    ret = spo.fmin_l_bfgs_b(self.grad_optimize_ei_over_hypers,
            #                            cand2[i,:].flatten(),
            #                            args=(comp,pend,vals,labels,True),
            #                            bounds=b, disp=0)
            #    cand2[i,:] = ret[0]

            cand = np.vstack((cand, cand2))

            overall_ei = self.ei_over_hypers(comp,pend,cand,vals,labels)
            best_cand = np.argmax(np.mean(overall_ei, axis=1))

            self.dump_hypers()
            if (best_cand >= numcand):
                return (int(numcand), cand[best_cand,:])

            return int(candidates[best_cand])

        else:
            print ('This Chooser module permits only slice sampling with > 0 '
                   'samples.')
            raise Exception('mcmc_iters <= 0')

    # Predict constraint voilating points
    def pred_constraint_voilation(self, cand, comp, vals):
        # The primary covariances for prediction.
        comp_cov   = self.cov(self.constraint_amp2, self.constraint_ls, comp)
        cand_cross = self.cov(self.constraint_amp2, self.constraint_ls, comp, 
                              cand)

        # Compute the required Cholesky.
        obsv_cov  = comp_cov + self.constraint_noise*np.eye(comp.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)

        cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
        cand_cross_grad = cov_grad_func(self.constraint_ls, comp, cand)

        # Predictive things.
        # Solve the linear systems.
        alpha  = spla.cho_solve((obsv_chol, True), self.ff)
        beta   = spla.solve_triangular(obsv_chol, cand_cross, lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha)# + self.constraint_mean
        func_m = sps.norm.cdf(func_m*self.constraint_gain)

        return func_m

    # Compute EI over hyperparameter samples
    def ei_over_hypers(self,comp,pend,cand,vals,labels):
        overall_ei = np.zeros((cand.shape[0], self.mcmc_iters))
        for mcmc_iter in xrange(self.mcmc_iters):
            hyper = self.hyper_samples[mcmc_iter]
            constraint_hyper = self.constraint_hyper_samples[mcmc_iter]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]

            self.constraint_mean = constraint_hyper[0]
            self.constraint_gain = constraint_hyper[1]
            self.constraint_amp2 = constraint_hyper[2]
            self.constraint_ls = constraint_hyper[3]
            overall_ei[:,mcmc_iter] = self.compute_constrained_ei(comp, pend, 
                                                                  cand, vals, 
                                                                  labels)
            
        return overall_ei

    # Adjust points by optimizing EI over a set of hyperparameter samples
    def grad_optimize_ei_over_hypers(self, cand, comp, pend, vals, labels, 
                                     compute_grad=True):
        summed_ei = 0
        summed_grad_ei = np.zeros(cand.shape).flatten()
        
        for mcmc_iter in xrange(self.mcmc_iters):
            hyper = self.hyper_samples[mcmc_iter]
            constraint_hyper = self.constraint_hyper_samples[mcmc_iter]
            self.mean = hyper[0]
            self.noise = hyper[1]
            self.amp2 = hyper[2]
            self.ls = hyper[3]

            self.constraint_mean = constraint_hyper[0]
            self.constraint_gain = constraint_hyper[1]
            self.constraint_amp2 = constraint_hyper[2]
            self.constraint_ls = constraint_hyper[3]
            if compute_grad:
                (ei,g_ei) = self.grad_optimize_ei(cand, comp, pend, vals, labels,
                                                  compute_grad)
                summed_grad_ei = summed_grad_ei + g_ei
            else:
                ei = self.grad_optimize_ei(cand, comp, pend, vals, 
                                           labels, compute_grad)
                
            summed_ei += ei

        if compute_grad:
            return (summed_ei, summed_grad_ei)
        else:
            return summed_ei

    def check_grad_ei(self, cand, comp, pend, vals, labels):
        (ei,dx1) = self.grad_optimize_ei_over_hypers(cand, comp, pend, vals, labels)
        dx2 = dx1*0
        idx = np.zeros(cand.shape[0])
        for i in xrange(0, cand.shape[0]):
            idx[i] = 1e-6
            (ei1,tmp) = self.grad_optimize_ei_over_hypers(
                cand + idx, comp, pend, vals, labels)
            (ei2,tmp) = self.grad_optimize_ei_over_hypers(
                cand - idx, comp, pend, vals, labels)
            dx2[i] = (ei - ei2)/(2*1e-6)
            idx[i] = 0
        print 'computed grads', dx1
        print 'finite diffs', dx2
        print (dx1/dx2)
        print np.sum((dx1 - dx2)**2)
        time.sleep(2)

    def grad_optimize_ei(self, cand, comp, pend, vals, labels, compute_grad=True):        
        if pend.shape[0] == 0:            
            return self.grad_optimize_ei_nopend(cand, comp, vals, labels, 
                                                compute_grad=True)
        else:
            return self.grad_optimize_ei_pend(cand, comp, pend, vals, labels, 
                                              compute_grad=True)

    def grad_optimize_ei_pend(self, cand, comp, pend, vals, labels, compute_grad=True):
        # Here we have to compute the gradients for constrained ei
        # This means deriving through the two kernels, the one for predicting
        # constraint violations and the one predicting ei

        # First pull out violating points
        compfull = comp.copy()
        comp = comp[labels > 0, :]
        vals = vals[labels > 0]

        # Use standard EI if there aren't enough observations of either
        # positive or negative constraint violations
        use_vanilla_ei = (np.all(labels > 0) or np.all(labels <= 0))

        best = np.min(vals)
        cand = np.reshape(cand, (-1, comp.shape[1]))
        func_constraint_m = 1

        if (not use_vanilla_ei):

            # First we make predictions for the durations
            # Compute covariances
            comp_constraint_cov   = self.cov(self.constraint_amp2,
                                             self.constraint_ls,
                                             compfull)
            cand_constraint_cross = self.cov(self.constraint_amp2,
                                             self.constraint_ls,
                                             compfull,cand)

            # Cholesky decompositions
            obsv_constraint_cov  = (comp_constraint_cov +
                 self.constraint_noise*np.eye(compfull.shape[0]))
            obsv_constraint_chol = spla.cholesky(obsv_constraint_cov,lower=True)

            # Linear systems
            t_alpha  = spla.cho_solve((obsv_constraint_chol, True), self.ff)

            # Predict marginal mean times and (possibly) variances
            ff = np.dot(cand_constraint_cross.T, t_alpha)

            # Squash through Gaussian cdf
            func_constraint_m = sps.norm.cdf(self.constraint_gain*ff)

            # Apply covariance function
            cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
            cand_cross_grad = cov_grad_func(self.constraint_ls, compfull, cand)
            grad_cross_t = np.squeeze(cand_cross_grad)

        # Now compute the gradients w.r.t. ei
        # The primary covariances for prediction.
        comp_cov   = self.cov(self.amp2, self.ls, comp)
        cand_cross = self.cov(self.amp2, self.ls, comp, cand)
        comp_cov_full   = self.cov(self.amp2, self.ls, compfull)
        cand_cross_full = self.cov(self.amp2, self.ls, compfull, cand)

        # Create a composite vector of complete and pending.
        comp_pend = np.concatenate((comp, pend))

        # Compute the covariance and Cholesky decomposition.
        comp_pend_cov  = (self.cov(self.amp2, self.ls, comp_pend) +
                          self.noise*np.eye(comp_pend.shape[0]))
        comp_pend_chol = spla.cholesky(comp_pend_cov, lower=True)

        # Compute submatrices.
        pend_cross = self.cov(self.amp2, self.ls, comp, pend)
        pend_kappa = self.cov(self.amp2,self.ls, pend)

        # Use the sub-Cholesky.
        obsv_chol = comp_pend_chol[:comp.shape[0],:comp.shape[0]]

        # Compute the required Cholesky.
        #obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
        #obsv_chol = spla.cholesky(obsv_cov, lower=True)
        obsv_cov_full  = comp_cov_full + self.noise*np.eye(compfull.shape[0])
        obsv_chol_full = spla.cholesky( obsv_cov_full, lower=True)

        # Predictive things.
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
        pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0],self.pending_samples)) + self.mean

        # Include the fantasies.
        fant_vals = np.concatenate(
            (np.tile(vals[:,np.newaxis],
                     (1,self.pending_samples)), pend_fant))

        # Compute bests over the fantasies.
        bests = np.min(fant_vals, axis=0)

        # Now generalize from these fantasies.
        cand_cross = self.cov(self.amp2, self.ls, comp_pend, cand)
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

        #beta  = spla.solve_triangular(obsv_chol_full, cand_cross_full, 
        #                               lower=True)
        #beta   = spla.solve_triangular(obsv_chol, cand_cross, 
        #                               lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)
        
        # Expected improvement
        func_s = np.sqrt(func_v)
        u      = (best - func_m) / func_s
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = func_s*(u*ncdf + npdf)

        constrained_ei = -np.sum(ei*func_constraint_m)
        if not compute_grad:
            return constrained_ei

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5*npdf / func_s

        # Apply covariance function        
        grad_cross = np.squeeze(cand_cross_grad)
        
        grad_xp_m = np.dot(alpha.transpose(),grad_cross)
        grad_xp_v = np.dot(-2*spla.cho_solve(
                (comp_pend_chol, True),cand_cross).transpose(), grad_cross)

        grad_xp = 0.5*self.amp2*(grad_xp_m*np.tile(g_ei_m,(comp.shape[1],1)).T + (grad_xp_v.T*g_ei_s2).T)
        
        grad_xp = np.sum(grad_xp,axis=0)

        if use_vanilla_ei:
            return -np.sum(ei), grad_xp.flatten()

        grad_constraint_xp_m = np.dot(t_alpha.transpose(),grad_cross_t)
        grad_constraint_xp_m = (0.5*self.constraint_amp2*
                                self.constraint_gain*
                                grad_constraint_xp_m*
                                sps.norm.pdf(self.constraint_gain*ff))

        grad_xp = (func_constraint_m*grad_xp + np.sum(ei)*grad_constraint_xp_m)

        return constrained_ei, grad_xp.flatten()

    def grad_optimize_ei_nopend(self, cand, comp, vals, labels, compute_grad=True):
        # Here we have to compute the gradients for constrained ei
        # This means deriving through the two kernels, the one for predicting
        # constraint violations and the one predicting ei

        # First pull out violating points
        compfull = comp.copy()
        comp = comp[labels > 0, :]
        vals = vals[labels > 0]

        # Use standard EI if there aren't enough observations of either
        # positive or negative constraint violations
        use_vanilla_ei = (np.all(labels > 0) or np.all(labels <= 0))

        best = np.min(vals)
        cand = np.reshape(cand, (-1, comp.shape[1]))
        func_constraint_m = 1

        if (not use_vanilla_ei):

            # First we make predictions for the durations
            # Compute covariances
            comp_constraint_cov   = self.cov(self.constraint_amp2,
                                             self.constraint_ls,
                                             compfull)
            cand_constraint_cross = self.cov(self.constraint_amp2,
                                             self.constraint_ls,
                                             compfull,cand)

            # Cholesky decompositions
            obsv_constraint_cov  = (comp_constraint_cov +
                 self.constraint_noise*np.eye(compfull.shape[0]))
            obsv_constraint_chol = spla.cholesky(obsv_constraint_cov,lower=True)

            # Linear systems
            t_alpha  = spla.cho_solve((obsv_constraint_chol, True), self.ff)

            # Predict marginal mean times and (possibly) variances
            ff = np.dot(cand_constraint_cross.T, t_alpha)

            # Squash through Gaussian cdf
            func_constraint_m = sps.norm.cdf(self.constraint_gain*ff)

        # Now compute the gradients w.r.t. ei
        # The primary covariances for prediction.
        comp_cov   = self.cov(self.amp2, self.ls, comp)
        cand_cross = self.cov(self.amp2, self.ls, comp, cand)
        comp_cov_full   = self.cov(self.amp2, self.ls, compfull)
        cand_cross_full = self.cov(self.amp2, self.ls, compfull, cand)

        # Compute the required Cholesky.
        obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
        obsv_chol = spla.cholesky(obsv_cov, lower=True)
        obsv_cov_full  = comp_cov_full + self.noise*np.eye(compfull.shape[0])
        obsv_chol_full = spla.cholesky( obsv_cov_full, lower=True)

        # Predictive things.
        # Solve the linear systems.
        alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
        beta   = spla.solve_triangular(obsv_chol_full, cand_cross_full, 
                                       lower=True)

        # Predict the marginal means and variances at candidates.
        func_m = np.dot(cand_cross.T, alpha) + self.mean
        func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)
        
        # Expected improvement
        func_s = np.sqrt(func_v)
        u      = (best - func_m) / func_s
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = func_s*(u*ncdf + npdf)

        constrained_ei = -np.sum(ei*func_constraint_m)
        if not compute_grad:
            return constrained_ei

        # Gradients of ei w.r.t. mean and variance
        g_ei_m = -ncdf
        g_ei_s2 = 0.5*npdf / func_s

        # Apply covariance function
        cov_grad_func = getattr(gp, 'grad_' + self.cov_func.__name__)
        cand_cross_grad = cov_grad_func(self.ls, comp, cand)
        grad_cross = np.squeeze(cand_cross_grad)

        cand_cross_grad_full = cov_grad_func(self.ls, compfull, cand)
        grad_cross_full = np.squeeze(cand_cross_grad_full)
        
        grad_xp_m = np.dot(alpha.transpose(),grad_cross)
        grad_xp_v = np.dot(-2*spla.cho_solve((obsv_chol_full, True),
                                             cand_cross_full).transpose(),
                            grad_cross_full)
        
        grad_xp = 0.5*self.amp2*(grad_xp_m*g_ei_m + grad_xp_v*g_ei_s2)

        if use_vanilla_ei:
            return -np.sum(ei), grad_xp.flatten()

        # Apply constraint classifier
        cand_cross_grad = cov_grad_func(self.constraint_ls, compfull, cand)
        grad_cross_t = np.squeeze(cand_cross_grad)

        grad_constraint_xp_m = np.dot(t_alpha.transpose(),grad_cross_t)
        grad_constraint_xp_m = (0.5*self.constraint_amp2*
                                self.constraint_gain*
                                grad_constraint_xp_m*
                                sps.norm.pdf(self.constraint_gain*ff))

        grad_xp = (func_constraint_m*grad_xp + ei*grad_constraint_xp_m)

        return constrained_ei, grad_xp.flatten()


    def compute_constrained_ei(self, comp, pend, cand, vals, labels):
        # First we make predictions for the durations as that
        # doesn't depend on pending experiments
        # First pull out violating points
        compfull = comp.copy()
        comp = comp[labels > 0, :]
        vals = vals[labels > 0]

        # Use standard EI if there aren't enough observations of either 
        # positive or negative constraint violations
        if (np.all(labels > 0) or np.all(labels <= 0)):
            func_constraint_m = 1
        else:
            # Compute covariances
            comp_constraint_cov   = self.cov(self.constraint_amp2, 
                                             self.constraint_ls, 
                                             compfull)
            cand_constraint_cross = self.cov(self.constraint_amp2, 
                                             self.constraint_ls,
                                             compfull,cand)

            # Cholesky decompositions
            obsv_constraint_cov  = (comp_constraint_cov + 
                self.constraint_noise*np.eye(compfull.shape[0]))
            obsv_constraint_chol = spla.cholesky(
                obsv_constraint_cov, lower=True)

            # Linear systems
            t_alpha  = spla.cho_solve((obsv_constraint_chol, True), self.ff)
            t_beta   = spla.solve_triangular(obsv_constraint_chol, 
                                             cand_constraint_cross, lower=True)

            # Predict marginal mean times and (possibly) variances
            func_constraint_m = (np.dot(cand_constraint_cross.T, t_alpha))

        # Squash through a probit
        func_constraint_m = sps.norm.cdf(self.constraint_gain*func_constraint_m)
        if pend.shape[0] == 0:
            # If there are no pending, don't do anything fancy.
            # Current best.
            best = np.min(vals)

            # The primary covariances for prediction.
            comp_cov   = self.cov(self.amp2, self.ls, comp)
            comp_cov_full = self.cov(self.amp2, self.ls, compfull)
            cand_cross = self.cov(self.amp2, self.ls, comp, cand)
            cand_cross_full = self.cov(self.amp2, self.ls, compfull, cand)

            # Compute the required Cholesky.
            obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
            obsv_cov_full  = (comp_cov_full + 
                              self.noise*np.eye(compfull.shape[0]))
            obsv_chol = spla.cholesky( obsv_cov, lower=True)
            obsv_chol_full = spla.cholesky( obsv_cov_full, lower=True)

            # Solve the linear systems.
            alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
            beta   = spla.solve_triangular(obsv_chol, cand_cross, lower=True)
            #beta   = spla.solve_triangular(obsv_chol_full, cand_cross_full,
            #                               lower=True)
            
            # Predict the marginal means and variances at candidates.
            func_m = np.dot(cand_cross.T, alpha) + self.mean
            func_v = self.amp2*(1+1e-6) - np.sum(beta**2, axis=0)

            # Expected improvement
            func_s = np.sqrt(func_v)
            u      = (best - func_m) / func_s
            ncdf   = sps.norm.cdf(u)
            npdf   = sps.norm.pdf(u)
            ei     = func_s*( u*ncdf + npdf)

            constrained_ei = ei*func_constraint_m
            return constrained_ei
        else:
            # If there are pending experiments, fantasize their outcomes.
            
            # Create a composite vector of complete and pending.
            comp_pend = np.concatenate((comp, pend))

            # Compute the covariance and Cholesky decomposition.
            comp_pend_cov  = (self.cov(self.amp2, self.ls, comp_pend) + 
                              self.noise*np.eye(comp_pend.shape[0]))
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
            pend_fant = np.dot(pend_chol, npr.randn(pend.shape[0],
                               self.pending_samples)) + self.mean

            # Include the fantasies.
            fant_vals = np.concatenate((np.tile(vals[:,np.newaxis], 
                                       (1,self.pending_samples)), pend_fant))

            # Compute bests over the fantasies.
            bests = np.min(fant_vals, axis=0)

            # Now generalize from these fantasies.
            cand_cross = self.cov(self.amp2, self.ls, comp_pend, cand)

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

            return np.mean(ei, axis=1)*func_constraint_m

    def compute_ei(self, comp, pend, cand, vals, labels):
        # First we make predictions for the durations as that
        # doesn't depend on pending experiments
        # First pull out violating points
        compfull = comp.copy()
        comp = comp[labels > 0, :]
        vals = vals[labels > 0]

        # Compute covariances
        comp_constraint_cov   = self.cov(self.constraint_amp2, 
                                         self.constraint_ls, 
                                         compfull)
        cand_constraint_cross = self.cov(self.constraint_amp2, 
                                         self.constraint_ls,
                                         compfull,cand)

        # Cholesky decompositions
        obsv_constraint_cov  = (comp_constraint_cov + 
                                self.constraint_noise*np.eye(
            compfull.shape[0]))
        obsv_constraint_chol = spla.cholesky( obsv_constraint_cov, lower=True )

        # Linear systems
        t_alpha  = spla.cho_solve((obsv_constraint_chol, True), self.ff)

        # Predict marginal mean times and (possibly) variances
        func_constraint_m = (np.dot(cand_constraint_cross.T, t_alpha))

        # Squash through a probit to get prob of not violating a constraint
        func_constraint_m = 1./(1+np.exp(-self.constraint_gain*
                                          func_constraint_m))
        
        if pend.shape[0] == 0:
            # If there are no pending, don't do anything fancy.

            # Current best.
            best = np.min(vals)

            # The primary covariances for prediction.
            comp_cov   = self.cov(self.amp2, self.ls, comp)
            comp_cov_full = self.cov(self.amp2, self.ls, compfull)
            cand_cross = self.cov(self.amp2, self.ls, comp, cand)
            cand_cross_full = self.cov(self.amp2, self.ls, compfull, cand)

            # Compute the required Cholesky.
            obsv_cov  = comp_cov + self.noise*np.eye(comp.shape[0])
            obsv_cov_full  = (comp_cov_full + 
                              self.noise*np.eye(compfull.shape[0]))
            obsv_chol = spla.cholesky( obsv_cov, lower=True )
            obsv_chol_full = spla.cholesky( obsv_cov_full, lower=True )

            # Solve the linear systems.
            alpha  = spla.cho_solve((obsv_chol, True), vals - self.mean)
            beta   = spla.solve_triangular(obsv_chol, cand_cross, lower=True)
            #beta   = spla.solve_triangular(obsv_chol_full, cand_cross_full, 
                                #lower=True)
            
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
            return 0

    def sample_constraint_hypers(self, comp, labels):
        # The latent GP projection
        # The latent GP projection
        if (self.ff is None or self.ff.shape[0] < comp.shape[0]):
            self.ff_samples = []
            comp_cov  = self.cov(self.constraint_amp2, self.constraint_ls, comp)
            obsv_cov  = comp_cov + 1e-6*np.eye(comp.shape[0])
            obsv_chol = spla.cholesky(obsv_cov, lower=True)
            self.ff = np.dot(obsv_chol,npr.randn(obsv_chol.shape[0]))

        self._sample_constraint_noisy(comp, labels)
        self._sample_constraint_ls(comp, labels)
        self.constraint_hyper_samples.append((self.constraint_mean, 
                                              self.constraint_gain, 
                                              self.constraint_amp2,
                                              self.constraint_ls))
        self.ff_samples.append(self.ff)

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
                                  1e-6*np.eye(comp.shape[0])) + 
                     self.noise*np.eye(comp.shape[0]))
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - self.mean)
            lp    = (-np.sum(np.log(np.diag(chol))) - 
                      0.5*np.dot(vals-self.mean, solve))
            return lp

        self.ls = util.slice_sample(self.ls, logprob, compwise=True)

    def _sample_constraint_ls(self, comp, vals):
        def lpProbit(ff, gain=self.constraint_gain):
            probs = sps.norm.cdf(ff*gain)
            probs[probs <= 0] = 1e-12
            probs[probs >= 1] = 1-1e-12
            llh = np.sum(vals*np.log(probs) + 
                         (1-vals)*np.log(1-probs))

            return llh

        def lpSigmoid(ff, gain=self.constraint_gain):
            probs = 1./(1. + np.exp(-gain*ff));
            probs[probs <= 0] = 1e-12
            probs[probs >= 1] = 1-1e-12
            llh   = np.sum(vals*np.log(probs) + (1-vals)*np.log(1-probs));
            return llh

        def updateGain(gain):
            if gain < 0.01 or gain > 10:
                return -np.inf

            cov   = (self.constraint_amp2 * (self.cov_func(
                        self.constraint_ls, comp, None) + 
                                             1e-6*np.eye(comp.shape[0])) + 
                     self.constraint_noise*np.eye(comp.shape[0]))
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals)
            lp   = lpProbit(self.ff, gain)

            return lp

        def logprob(ls):
            if np.any(ls < 0) or np.any(ls > self.constraint_max_ls):
                return -np.inf
        
            cov   = self.constraint_amp2 * (self.cov_func(ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + self.constraint_noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), self.ff)
            lp   = lpProbit(self.ff)

            return lp

        hypers = util.slice_sample(self.constraint_ls, logprob, compwise=True)
        self.constraint_ls = hypers

        cov   = self.constraint_amp2 * (self.cov_func(self.constraint_ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + self.constraint_noise*np.eye(comp.shape[0])
        chol  = spla.cholesky(cov, lower=False)
        ff = self.ff
        for jj in xrange(20):
            (ff, lpell) = self.elliptical_slice(ff, chol, lpProbit)
            
        self.ff = ff

        # Update gain
        hypers = util.slice_sample(np.array([self.constraint_gain]),
                                   updateGain, compwise=True)
        self.constraint_gain = hypers[0]

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
            
            cov   = amp2 * ((self.cov_func(self.ls, comp, None) + 
                            1e-6*np.eye(comp.shape[0])) + 
                            noise*np.eye(comp.shape[0]))
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)

            # Roll in noise horseshoe prior.
            lp += np.log(np.log(1 + (self.noise_scale/noise)**2))

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.amp2_scale)**2

            return lp

        hypers = util.slice_sample(np.array(
                    [self.mean, self.amp2, self.noise]), 
                                   logprob, compwise=False)
        self.mean  = hypers[0]
        self.amp2  = hypers[1]
        self.noise = hypers[2]

    def _sample_constraint_noisy(self, comp, vals):
        def lpProbit(ff, gain=self.constraint_gain):
            probs = sps.norm.cdf(ff*gain)        
            probs[probs <= 0] = 1e-12
            probs[probs >= 1] = 1-1e-12
            llh = np.sum(vals*np.log(probs) + 
                         (1-vals)*np.log(1-probs))
            if np.any(np.isnan(probs)):
                print probs
            return llh

        def lpSigmoid(ff,gain=self.constraint_gain):
            probs = 1./(1. + np.exp(-gain*ff));
            probs[probs <= 0] = 1e-12
            probs[probs >= 1] = 1-1e-12
            llh   = np.sum(vals*np.log(probs) + (1-vals)*np.log(1-probs));
            return llh

        def logprob(hypers):
            amp2  = hypers[0]
            ff = hypers[1:]

            if amp2 < 0:
                return -np.inf

            noise = self.constraint_noise
            cov   = amp2 * (self.cov_func(self.constraint_ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), ff)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(ff, solve)

            # Roll in amplitude lognormal prior
            lp -= 0.5*(np.log(amp2)/self.constraint_amp2_scale)**2

            lp   += lpProbit(ff,self.constraint_gain)

            return lp

        hypers = util.slice_sample(np.hstack((np.array([self.constraint_amp2]),
                                            self.ff)), logprob, compwise=False)
        self.constraint_amp2  = hypers[0]
        self.ff = hypers[1:]
        cov   = self.constraint_amp2 * ((
                    self.cov_func(self.constraint_ls, comp, None) + 
                    1e-6*np.eye(comp.shape[0])) + 
                                self.constraint_noise*np.eye(comp.shape[0]))
        chol  = spla.cholesky(cov, lower=False)
        ff = self.ff
        for jj in xrange(50):
            (ff, lpell) = self.elliptical_slice(ff, chol, lpProbit)
        self.ff = ff

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
            
            cov   = amp2 * ((self.cov_func(self.ls, comp, None) + 
                             1e-6*np.eye(comp.shape[0])) + 
                            noise*np.eye(comp.shape[0]))
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

    def elliptical_slice(self, xx, chol_Sigma, log_like_fn, cur_log_like=None, 
                         angle_range=0):
        D = xx.shape[0]

        if cur_log_like is None:
            cur_log_like = log_like_fn(xx)

        nu = np.dot(chol_Sigma.T,np.random.randn(D, 1)).flatten()
        hh = np.log(np.random.rand()) + cur_log_like
        
        # Set up a bracket of angles and pick a first proposal.
        # "phi = (theta'-theta)" is a change in angle.
        if angle_range <= 0:
            # Bracket whole ellipse with both edges at first proposed point
            phi = np.random.rand()*2*math.pi;
            phi_min = phi - 2*math.pi;
            phi_max = phi;
        else:
            # Randomly center bracket on current point
            phi_min = -angle_range*np.random.rand();
            phi_max = phi_min + angle_range;
            phi = np.random.rand()*(phi_max - phi_min) + phi_min;

        # Slice sampling loop
        while True:
            # Compute xx for proposed angle difference 
            # and check if it's on the slice
            xx_prop = xx*np.cos(phi) + nu*np.sin(phi);

            cur_log_like = log_like_fn(xx_prop);
            if cur_log_like > hh:
                # New point is on slice, ** EXIT LOOP **
                break;

            # Shrink slice to rejected point
            if phi > 0:
                phi_max = phi;
            elif phi < 0:
                phi_min = phi;
            else:
                raise Exception('BUG DETECTED: Shrunk to current position '
                                'and still not acceptable.');

            # Propose new angle difference
            phi = np.random.rand()*(phi_max - phi_min) + phi_min;

        xx = xx_prop;
        return (xx, cur_log_like)
