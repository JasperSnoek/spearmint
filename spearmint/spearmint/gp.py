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
"""
gp.py contains utility functions related to computation in Gaussian processes.
"""
import numpy as np
import scipy.linalg as spla
import scipy.optimize as spo
import scipy.io as sio
import scipy.weave
    
SQRT_3 = np.sqrt(3.0)
SQRT_5 = np.sqrt(5.0)

def dist2(ls, x1, x2=None):
    # Assumes NxD and MxD matrices.
    # Compute the squared distance matrix, given length scales.
    
    if x2 is None:
        # Find distance with self for x1.

        # Rescale.
        xx1 = x1 / ls        
        xx2 = xx1

    else:
        # Rescale.
        xx1 = x1 / ls
        xx2 = x2 / ls
    
    r2 = np.maximum(-(np.dot(xx1, 2*xx2.T) 
                       - np.sum(xx1*xx1, axis=1)[:,np.newaxis]
                       - np.sum(xx2*xx2, axis=1)[:,np.newaxis].T), 0.0)

    return r2

def grad_dist2(ls, x1, x2=None):
    if x2 is None:
        x2 = x1
        
    # Rescale.
    x1 = x1 / ls
    x2 = x2 / ls
    
    N = x1.shape[0]
    M = x2.shape[0]
    D = x1.shape[1]
    gX = np.zeros((x1.shape[0],x2.shape[0],x1.shape[1]))

    code = \
    """
    for (int i=0; i<N; i++)
      for (int j=0; j<M; j++)
        for (int d=0; d<D; d++)
          gX(i,j,d) = (2/ls(d))*(x1(i,d) - x2(j,d));
    """
    try:
        scipy.weave.inline(code, ['x1','x2','gX','ls','M','N','D'], \
                       type_converters=scipy.weave.converters.blitz, \
                       compiler='gcc')
    except:
        # The C code weave above is 10x faster than this:
        for i in xrange(0,x1.shape[0]):
            gX[i,:,:] = 2*(x1[i,:] - x2[:,:])*(1/ls)

    return gX

def SE(ls, x1, x2=None, grad=False):
    ls = np.ones(ls.shape)
    cov = np.exp(-0.5 * dist2(ls, x1, x2))
    if grad:
        return (cov, grad_ARDSE(ls, x1, x2))
    else:
        return cov

def ARDSE(ls, x1, x2=None, grad=False):
    cov = np.exp(-0.5 * dist2(ls, x1, x2))
    if grad:
        return (cov, grad_ARDSE(ls, x1, x2))
    else:
        return cov

def grad_ARDSE(ls, x1, x2=None):
    r2 = dist2(ls, x1, x2)
    r  = np.sqrt(r2)
    return -0.5*np.exp(-0.5*r2)[:,:,np.newaxis] * grad_dist2(ls, x1, x2)

def Matern32(ls, x1, x2=None, grad=False):
    r   = np.sqrt(dist2(ls, x1, x2))
    cov = (1 + SQRT_3*r) * np.exp(-SQRT_3*r)
    if grad:
        return (cov, grad_Matern32(ls, x1, x2))
    else:
        return cov

def grad_Matern32(ls, x1, x2=None):
    r       = np.sqrt(dist2(ls, x1, x2))
    grad_r2 = -1.5*np.exp(-SQRT_3*r)
    return grad_r2[:,:,np.newaxis] * grad_dist2(ls, x1, x2)

def Matern52(ls, x1, x2=None, grad=False):
    r2  = np.abs(dist2(ls, x1, x2))
    r   = np.sqrt(r2)
    cov = (1.0 + SQRT_5*r + (5.0/3.0)*r2) * np.exp(-SQRT_5*r)
    if grad:
        return (cov, grad_Matern52(ls, x1, x2))
    else:
        return cov

def grad_Matern52(ls, x1, x2=None):
    r       = np.sqrt(dist2(ls, x1, x2))
    grad_r2 = -(5.0/6.0)*np.exp(-SQRT_5*r)*(1 + SQRT_5*r)
    return grad_r2[:,:,np.newaxis] * grad_dist2(ls, x1, x2)

class GP:
    def __init__(self, covar="Matern52", mcmc_iters=10, noiseless=False):
        self.cov_func        = globals()[covar]
        self.mcmc_iters      = int(mcmc_iters)
        self.D               = -1
        self.hyper_iters     = 1
        self.noiseless       = bool(int(noiseless))
        self.hyper_samples = []
        
        self.noise_scale = 0.1  # horseshoe prior 
        self.amp2_scale  = 1    # zero-mean log normal prior
        self.max_ls      = 2    # top-hat prior on length scales 

    def real_init(self, dims, values):
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

    def cov(self, x1, x2=None):
        if x2 is None:
            return self.amp2 * (self.cov_func(self.ls, x1, None)
                                + 1e-6*np.eye(x1.shape[0]))
        else:
            return self.amp2 * self.cov_func(self.ls, x1, x2)

    def logprob(self, comp, vals):
            mean  = self.mean
            amp2  = self.amp2
            noise = self.noise
            
            cov   = amp2 * (self.cov_func(self.ls, comp, None) + 1e-6*np.eye(comp.shape[0])) + noise*np.eye(comp.shape[0])
            chol  = spla.cholesky(cov, lower=True)
            solve = spla.cho_solve((chol, True), vals - mean)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(vals-mean, solve)
            return lp

    def optimize_hypers(self, comp, vals):
        self.mean = np.mean(vals)
        diffs     = vals - self.mean

        state = { }

        def jitter_chol(covmat):
            passed = False
            jitter = 1e-8
            val = 0
            while not passed:
                if (jitter > 100000):
                    val = spla.cholesky(np.eye(covmat.shape[0]))
                    break
                try:
                    val = spla.cholesky(covmat +
                        jitter*np.eye(covmat.shape[0]), lower=True)
                    passed = True
                except ValueError:
                    jitter = jitter*1.1
                    print "Covariance matrix not PSD, adding jitter:", jitter
                    passed = False
            return val
        
        def memoize(amp2, noise, ls):
            if ( 'corr' not in state
                 or state['amp2'] != amp2
                 or state['noise'] != noise
                 or np.any(state['ls'] != ls)):

                # Get the correlation matrix
                (corr, grad_corr) = self.cov_func(ls, comp, None, grad=True)
        
                # Scale and add noise & jitter.
                covmat = (amp2 * (corr + 1e-6*np.eye(comp.shape[0])) 
                          + noise * np.eye(comp.shape[0]))

                # Memoize
                state['corr']      = corr
                state['grad_corr'] = grad_corr
                state['chol']      = jitter_chol(covmat)
                state['amp2']      = amp2
                state['noise']     = noise
                state['ls']        = ls
                
            return (state['chol'], state['corr'], state['grad_corr'])

        def nlogprob(hypers):
            amp2  = np.exp(hypers[0])
            noise = np.exp(hypers[1])
            ls    = np.exp(hypers[2:])

            chol  = memoize(amp2, noise, ls)[0]
            solve = spla.cho_solve((chol, True), diffs)
            lp    = -np.sum(np.log(np.diag(chol)))-0.5*np.dot(diffs, solve)
            return -lp

        def grad_nlogprob(hypers):
            amp2  = np.exp(hypers[0])
            noise = np.exp(hypers[1])
            ls    = np.exp(hypers[2:])

            chol, corr, grad_corr = memoize(amp2, noise, ls)
            solve   = spla.cho_solve((chol, True), diffs)
            inv_cov = spla.cho_solve((chol, True), np.eye(chol.shape[0]))

            jacobian = np.outer(solve, solve) - inv_cov

            grad = np.zeros(self.D + 2)

            # Log amplitude gradient.
            grad[0] = 0.5 * np.trace(np.dot( jacobian, corr + 1e-6*np.eye(chol.shape[0]))) * amp2

            # Log noise gradient.
            grad[1] = 0.5 * np.trace(np.dot( jacobian, np.eye(chol.shape[0]))) * noise

            # Log length scale gradients.
            for dd in xrange(self.D):
                grad[dd+2] = 1 * np.trace(np.dot( jacobian, -amp2*grad_corr[:,:,dd]*comp[:,dd][:,np.newaxis]/(np.exp(ls[dd]))))*np.exp(ls[dd])

            # Roll in the prior variance.
            #grad -= 2*hypers/self.hyper_prior

            return -grad
        
        # Initial length scales.
        self.ls = np.ones(self.D)
        # Initial amplitude.
        self.amp2 = np.std(vals)
        # Initial observation noise.
        self.noise = 1e-3
        
        hypers     = np.zeros(self.ls.shape[0]+2)
        hypers[0]  = np.log(self.amp2)
        hypers[1]  = np.log(self.noise)
        hypers[2:] = np.log(self.ls)
        
        # Use a bounded bfgs just to prevent the length-scales and noise from 
        # getting into regions that are numerically unstable
        b = [(-10,10),(-10,10)]
        for i in xrange(comp.shape[1]):
            b.append((-10,5))
  
        hypers = spo.fmin_l_bfgs_b(nlogprob, hypers, grad_nlogprob, args=(), bounds=b, disp=0)
                
        #hypers = spo.fmin_bfgs(nlogprob, hypers, grad_nlogprob, maxiter=100)
        hypers = hypers[0]
        #hypers = spo.fmin_bfgs(nlogprob, hypers, grad_nlogprob, maxiter=100)

        self.amp2  = np.exp(hypers[0])
        self.noise = np.exp(hypers[1])
        self.ls    = np.exp(hypers[2:])

def main():
    try:
        import matplotlib.pyplot as plt
    except:
        pass

    # Let's start with some random values
    x = np.linspace(0,1,10)[:,np.newaxis]*10#np.random.rand(100)[:,np.newaxis]
    y = np.random.randn(10)
    mygp = GP(covar='ARDSE')
    mygp.real_init(x.shape[1], y)

    # Sample some functions given these hyperparameters and plot them
    for i in xrange(0,5):
        x = np.linspace(0,1,100)[:,np.newaxis]*10
        K = mygp.cov(x)
        y = np.random.randn(100)
    
        fsamp = mygp.mean + np.dot(spla.cholesky(K).transpose(), y)
        try:
            plt.plot(x, fsamp)
        except:
            pass

    print 'Loglikelihood before optimizing: ', mygp.logprob(x,y)
    mygp.optimize_hypers(x,y)
    print 'Loglikelihood after optimizing: ', mygp.logprob(x,y)
        
    try:
        plt.show()
    except:
        print 'Install matplotlib to get figures'        

if __name__ == '__main__':
    main()
