#!/usr/bin/env python
"""Module cma implements the CMA-ES, Covariance Matrix Adaptation Evolution
Strategy, a stochastic optimizer for robust non-linear non-convex
derivative-free function minimization for Python versions 2.6, 2.7, 3.x
(for Python 2.5 class SolutionDict would need to be re-implemented, because
it depends on collections.MutableMapping, since version 0.91.01).

CMA-ES searches for a minimizer (a solution x in R**n) of an
objective function f (cost function), such that f(x) is
minimal. Regarding f, only function values for candidate solutions
need to be available, gradients are not necessary. Even less
restrictive, only a passably reliable ranking of the candidate
solutions in each iteration is necessary, the function values
itself do not matter. Some termination criteria however depend
on actual f-values.

Two interfaces are provided:

  - function `fmin(func, x0, sigma0,...)`
        runs a complete minimization
        of the objective function func with CMA-ES.

  - class `CMAEvolutionStrategy`
      allows for minimization such that the
      control of the iteration loop remains with the user.


Used packages:

    - unavoidable: `numpy` (see `barecmaes2.py` if `numpy` is not
      available),
    - avoidable with small changes: `time`, `sys`
    - optional: `matplotlib.pylab` (for `plot` etc., highly
      recommended), `pprint` (pretty print), `pickle` (in class
      `Sections`), `doctest`, `inspect`, `pygsl` (never by default)

Testing
-------
The code can be tested on a given system. Typing::

    python cma.py --test

or in the Python shell ``ipython -pylab``::

    run cma.py --test

runs ``doctest.testmod(cma)`` showing only exceptions (and not the
tests that fail due to small differences in the output) and should
run without complaints in about under two minutes. On some systems,
the pop up windows must be closed manually to continue and finish
the test.

Install
-------
The code can be installed by::

    python cma.py --install

which solely calls the ``setup`` function from the ``distutils.core``
package for installation.

Example
-------
::

    import cma
    help(cma)  # "this" help message, use cma? in ipython
    help(cma.fmin)
    help(cma.CMAEvolutionStrategy)
    help(cma.Options)
    cma.Options('tol')  # display 'tolerance' termination options
    cma.Options('verb') # display verbosity options
    res = cma.fmin(cma.Fcts.tablet, 15 * [1], 1)
    res[0]  # best evaluated solution
    res[5]  # mean solution, presumably better with noise

:See: `fmin()`, `Options`, `CMAEvolutionStrategy`

:Author: Nikolaus Hansen, 2008-2012

:License: GPL 2 and 3

"""
from __future__ import division  # future is >= 3.0, this code has mainly been used with 2.6 & 2.7
from __future__ import with_statement  # only necessary for python 2.5 and not in heavy use
# from __future__ import collections.MutableMapping # does not exist in future, otherwise 2.5 would work
from __future__ import print_function  # for cross-checking, available from python 2.6
import sys
if sys.version.startswith('3'):  # in python 3.x
    xrange = range
    raw_input = input

__version__ = "0.92.04 $Revision: 3322 $ $Date: 2012-11-22 18:05:10 +0100 (Thu, 22 Nov 2012) $"
#    bash: svn propset svn:keywords 'Date Revision' cma.py

#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 2 or 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# for testing:
#   pyflakes cma.py   # finds bugs by static analysis
#   pychecker --limit 60 cma.py  # also executes, gives 60 warnings (all checked)
#   python cma.py -t -quiet # executes implemented tests based on doctest

# to create a html documentation file:
#    pydoc -w cma  # edit the header (remove local pointers)
#    epydoc cma.py  # comes close to javadoc but does not find the
#                   # links of function references etc
#    doxygen needs @package cma as first line in the module docstring
#       some things like class attributes are not interpreted correctly
#    sphinx: doc style of doc.python.org, could not make it work

# TODO: make those options that are only used in fmin an error in init of CMA, but still Options() should
#       work as input to CMA.
# TODO: add a default logger in CMAEvolutionStrategy, see fmin() and optimize() first
#        tell() should probably not add data, but optimize() should handle even an after_iteration_handler.
# TODO: CMAEvolutionStrategy(ones(10), 1).optimize(cma.fcts.elli)  # should work like fmin
#       one problem: the data logger is not default and seemingly cannot be attached in one line
# TODO: check combination of boundary handling and transformation: penalty must be computed
#       on gp.pheno(x_geno, bounds=None), but without bounds, check/remove usage of .geno everywhere
# TODO: check whether all new solutions are put into self.sent_solutions
# TODO: separate initialize==reset_state from __init__
# TODO: introduce Zpos == diffC which makes the code more consistent and the active update "exact"
# TODO: split tell into a variable transformation part and the "pure" functionality
#       usecase: es.tell_geno(X, [func(es.pheno(x)) for x in X])
#       genotypic repair is not part of tell_geno
# TODO: read settable "options" from a (properties) file, see myproperties.py
#
# typical parameters in scipy.optimize: disp, xtol, ftol, maxiter, maxfun, callback=None
#         maxfev, diag (A sequency of N positive entries that serve as
#                 scale factors for the variables.)
#           full_output -- non-zero to return all optional outputs.
#   If xtol < 0.0, xtol is set to sqrt(machine_precision)
#    'infot -- a dictionary of optional outputs with the keys:
#                      'nfev': the number of function calls...
#
#    see eg fmin_powell
# typical returns
#        x, f, dictionary d
#        (xopt, {fopt, gopt, Hopt, func_calls, grad_calls, warnflag}, <allvecs>)
#
# TODO: keep best ten solutions
# TODO: implement constraints handling
# TODO: option full_output -- non-zero to return all optional outputs.
# TODO: extend function unitdoctest, or use unittest?
# TODO: implement equal-fitness termination, covered by stagnation?
# TODO: apply style guide: no capitalizations!?
# TODO: check and test dispdata()
# TODO: eigh(): thorough testing would not hurt
#
# TODO (later): implement readSignals from a file like properties file (to be called after tell())

import time  # not really essential
import collections, numpy as np # arange, cos, size, eye, inf, dot, floor, outer, zeros, linalg.eigh, sort, argsort, random, ones,...
from numpy import inf, array, dot, exp, log, sqrt, sum   # to access the built-in sum fct:  __builtins__.sum or del sum removes the imported sum and recovers the shadowed
try:
    import matplotlib.pylab as pylab  # also: use ipython -pylab
    show = pylab.show
    savefig = pylab.savefig   # we would like to be able to use cma.savefig() etc
    closefig = pylab.close
except:
    pylab = None
    print('  Could not import matplotlib.pylab, therefore ``cma.plot()`` etc. is not available')
    def show():
        pass

__docformat__ = "reStructuredText"  # this hides some comments entirely?

sys.py3kwarning = True  # TODO: out-comment from version 2.6

# why not package math?

# TODO: check scitools.easyviz and how big the adaptation would be

# changes:
# 12/10/25: removed useless check_points from fmin interface
# 12/10/17: bug fix printing number of infeasible samples, moved not-in-use methods
#           timesCroot and divCroot to the right class
# 12/10/16 (0.92.00): various changes commit: bug bound[0] -> bounds[0], more_to_write fixed,
#   sigma_vec introduced, restart from elitist, trace normalization, max(mu,popsize/2)
#   is used for weight calculation.
# 12/07/23: (bug:) BoundPenalty.update respects now genotype-phenotype transformation
# 12/07/21: convert value True for noisehandling into 1 making the output compatible
# 12/01/30: class Solution and more old stuff removed r3101
# 12/01/29: class Solution is depreciated, GenoPheno and SolutionDict do the job (v0.91.00, r3100)
# 12/01/06: CMA_eigenmethod option now takes a function (integer still works)
# 11/09/30: flat fitness termination checks also history length
# 11/09/30: elitist option (using method clip_or_fit_solutions)
# 11/09/xx: method clip_or_fit_solutions for check_points option for all sorts of
#           injected or modified solutions and even reliable adaptive encoding
# 11/08/19: fixed: scaling and typical_x type clashes 1 vs array(1) vs ones(dim) vs dim * [1]
# 11/07/25: fixed: fmin wrote first and last line even with verb_log==0
#           fixed: method settableOptionsList, also renamed to versatileOptions
#           default seed depends on time now
# 11/07/xx (0.9.92): added: active CMA, selective mirrored sampling, noise/uncertainty handling
#           fixed: output argument ordering in fmin, print now only used as function
#           removed: parallel option in fmin
# 11/07/01: another try to get rid of the memory leak by replacing self.unrepaired = self[:]
# 11/07/01: major clean-up and reworking of abstract base classes and of the documentation,
#           also the return value of fmin changed and attribute stop is now a method.
# 11/04/22: bug-fix: option fixed_variables in combination with scaling
# 11/04/21: stopdict is not a copy anymore
# 11/04/15: option fixed_variables implemented
# 11/03/23: bug-fix boundary update was computed even without boundaries
# 11/03/12: bug-fix of variable annotation in plots
# 11/02/05: work around a memory leak in numpy
# 11/02/05: plotting routines improved
# 10/10/17: cleaning up, now version 0.9.30
# 10/10/17: bug-fix: return values of fmin now use phenotyp (relevant
#           if input scaling_of_variables is given)
# 08/10/01: option evalparallel introduced,
#           bug-fix for scaling being a vector
# 08/09/26: option CMAseparable becomes CMA_diagonal
# 08/10/18: some names change, test functions go into a class
# 08/10/24: more refactorizing
# 10/03/09: upper bound exp(min(1,...)) for step-size control


# TODO: this would define the visible interface
# __all__ = ['fmin', 'CMAEvolutionStrategy', 'plot', ...]
#


# emptysets = ('', (), [], {}) # array([]) does not work but also np.size(.) == 0
# "x in emptysets" cannot be well replaced by "not x"
# which is also True for array([]) and None, but also for 0 and False, and False for NaN

use_sent_solutions = True  # 5-30% CPU slower, particularly for large lambda, will be mandatory soon

#____________________________________________________________
#____________________________________________________________
#
def unitdoctest():
    """is used to describe test cases and might in future become helpful
    as an experimental tutorial as well. The main testing feature at the
    moment is by doctest with ``cma._test()`` or conveniently by
    ``python cma.py --test``. With the ``--verbose`` option added, the
    results will always slightly differ and many "failed" test cases
    might be reported.

    A simple first overall test:
        >>> import cma
        >>> res = cma.fmin(cma.fcts.elli, 3*[1], 1, CMA_diagonal=2, seed=1, verb_time=0)
        (3_w,7)-CMA-ES (mu_w=2.3,w_1=58%) in dimension 3 (seed=1)
           Covariance matrix is diagonal for 2 iterations (1/ccov=7.0)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1       7 1.453161670768570e+04 1.2e+00 1.08e+00  1e+00  1e+00
            2      14 3.281197961927601e+04 1.3e+00 1.22e+00  1e+00  2e+00
            3      21 1.082851071704020e+04 1.3e+00 1.24e+00  1e+00  2e+00
          100     700 8.544042012075362e+00 1.4e+02 3.18e-01  1e-03  2e-01
          200    1400 5.691152415221861e-12 1.0e+03 3.82e-05  1e-09  1e-06
          220    1540 3.890107746209078e-15 9.5e+02 4.56e-06  8e-11  7e-08
        termination on tolfun : 1e-11
        final/bestever f-value = 3.89010774621e-15 2.52273602735e-15
        mean solution:  [ -4.63614606e-08  -3.42761465e-10   1.59957987e-11]
        std deviation: [  6.96066282e-08   2.28704425e-09   7.63875911e-11]

    Test on the Rosenbrock function with 3 restarts. The first trial only
    finds the local optimum, which happens in about 20% of the cases.
        >>> import cma
        >>> res = cma.fmin(cma.fcts.rosen, 4*[-1],1, ftarget=1e-6, restarts=3, verb_time=0, verb_disp=500, seed=3)
        (4_w,8)-CMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=3)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1       8 4.875315645656848e+01 1.0e+00 8.43e-01  8e-01  8e-01
            2      16 1.662319948123120e+02 1.1e+00 7.67e-01  7e-01  8e-01
            3      24 6.747063604799602e+01 1.2e+00 7.08e-01  6e-01  7e-01
          184    1472 3.701428610430019e+00 4.3e+01 9.41e-07  3e-08  5e-08
        termination on tolfun : 1e-11
        final/bestever f-value = 3.70142861043 3.70142861043
        mean solution:  [-0.77565922  0.61309336  0.38206284  0.14597202]
        std deviation: [  2.54211502e-08   3.88803698e-08   4.74481641e-08   3.64398108e-08]
        (8_w,16)-CMA-ES (mu_w=4.8,w_1=32%) in dimension 4 (seed=4)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1    1489 2.011376859371495e+02 1.0e+00 8.90e-01  8e-01  9e-01
            2    1505 4.157106647905128e+01 1.1e+00 8.02e-01  7e-01  7e-01
            3    1521 3.548184889359060e+01 1.1e+00 1.02e+00  8e-01  1e+00
          111    3249 6.831867555502181e-07 5.1e+01 2.62e-02  2e-04  2e-03
        termination on ftarget : 1e-06
        final/bestever f-value = 6.8318675555e-07 1.18576673231e-07
        mean solution:  [ 0.99997004  0.99993938  0.99984868  0.99969505]
        std deviation: [ 0.00018973  0.00038006  0.00076479  0.00151402]
        >>> assert res[1] <= 1e-6

    Notice the different termination conditions. Termination on the target
    function value ftarget prevents further restarts.

    Test of scaling_of_variables option
        >>> import cma
        >>> opts = cma.Options()
        >>> opts['seed'] = 456
        >>> opts['verb_disp'] = 0
        >>> opts['CMA_active'] = 1
        >>> # rescaling of third variable: for searching in  roughly
        >>> #   x0 plus/minus 1e3*sigma0 (instead of plus/minus sigma0)
        >>> opts.scaling_of_variables = [1, 1, 1e3, 1]
        >>> res = cma.fmin(cma.fcts.rosen, 4 * [0.1], 0.1, **opts)
        termination on tolfun : 1e-11
        final/bestever f-value = 2.68096173031e-14 1.09714829146e-14
        mean solution:  [ 1.00000001  1.00000002  1.00000004  1.00000007]
        std deviation: [  3.00466854e-08   5.88400826e-08   1.18482371e-07   2.34837383e-07]

    The printed std deviations reflect the actual true value (not the one
    in the internal representation which would be different).
        >>> import cma
        >>> r = cma.fmin(cma.fcts.diffpow, 15 * [1], 1, CMA_dampsvec_fac=0.5, ftarget=1e-9)
        >>> assert(r[1] < 1e-9)
        >>> assert(r[2] < 13000)  # only passed with CMA_dampsvec_fac


	:See: cma.main(), cma._test()

    """

    pass


#____________________________________________________________
#____________________________________________________________
#
class BlancClass(object):
    """blanc container class for having a collection of attributes"""

#_____________________________________________________________________
#_____________________________________________________________________
#
class DerivedDictBase(collections.MutableMapping):
    """for conveniently adding features to a dictionary. The actual
    dictionary is in ``self.data``. Copy-paste
    and modify setitem, getitem, and delitem, if necessary"""
    def __init__(self, *args, **kwargs):
        # collections.MutableMapping.__init__(self)
        super(DerivedDictBase, self).__init__()
        # super(SolutionDict, self).__init__()  # the same
        self.data = dict(*args, **kwargs)
    def __len__(self):
        return len(self.data)
    def __contains__(self, value):
        return value in self.data
    def __iter__(self):
        return iter(self.data)
    def __setitem__(self, key, value):
        """defines self[key] = value"""
        self.data[key] = value
    def __getitem__(self, key):
        """defines self[key]"""
        return self.data[key]
    def __delitem__(self, key):
        del self.data[key]

class SolutionDict(DerivedDictBase):
    """dictionary with computation of an hash key for the inserted solutions and
    a stack of previously inserted same solutions.
    Each entry is meant to store additional information related to the solution.

        >>> import cma, numpy as np
        >>> d = cma.SolutionDict()
        >>> x = np.array([1,2,4])
        >>> d[x] = {'x': x, 'iteration': 1}
        >>> d.get(x) == (d[x] if d.key(x) in d.keys() else None)

    The last line is always true.

    TODO: data_with_same_key behaves like a stack (see setitem and delitem), but rather should behave like a queue?!
    A queue is less consistent with the operation self[key] = ..., if self.data_with_same_key[key] is not empty.

    """
    def __init__(self, *args, **kwargs):
        DerivedDictBase.__init__(self, *args, **kwargs)
        self.data_with_same_key = {}
    def key(self, x):
        try:
            return tuple(x)
        except TypeError:
            return x
    def __setitem__(self, key, value):
        """defines self[key] = value"""
        key = self.key(key)
        if key in self.data_with_same_key:
            self.data_with_same_key[key] += [self.data[key]]
        elif key in self.data:
            self.data_with_same_key[key] = [self.data[key]]
        self.data[key] = value
    def __getitem__(self, key):
        """defines self[key]"""
        return self.data[self.key(key)]
    def __delitem__(self, key):
        """remove only most current key-entry"""
        key = self.key(key)
        if key in self.data_with_same_key:
            if len(self.data_with_same_key[key]) == 1:
                self.data[key] = self.data_with_same_key.pop(key)[0]
            else:
                self.data[key] = self.data_with_same_key[key].pop(-1)
        else:
            del self.data[key]
    def truncate(self, max_len, min_iter):
        if len(self) > max_len:
            for k in list(self.keys()):
                if self[k]['iteration'] < min_iter:
                    del self[k]  # only deletes one item with k as key, should delete all?

class SolutionDictOld(dict):
    """depreciated, SolutionDict should do, to be removed after SolutionDict
    has been successfully applied.
    dictionary with computation of an hash key for the inserted solutions and
    stack of previously inserted same solutions.
    Each entry is meant to store additional information related to the solution.
    Methods ``pop`` and ``get`` are modified accordingly.

        d = SolutionDict()
        x = array([1,2,4])
        d.insert(x, {'x': x, 'iteration': 1})
        d.get(x) == d[d.key(x)] if d.key(x) in d.keys() else d.get(x) is None

    TODO: not yet tested
    TODO: behaves like a stack (see _pop_derived), but rather should behave like a queue?!
    A queue is less consistent with the operation self[key] = ..., if self.more[key] is not empty.

    """
    def __init__(self):
        self.more = {}  # previously inserted same solutions
        self._pop_base = self.pop
        self.pop = self._pop_derived
        self._get_base = self.get
        self.get = self._get_derived
    def key(self, x):
        """compute the hash key of ``x``"""
        return tuple(x)
    def insert(self, x, datadict):
        key = self.key(x)
        if key in self.more:
            self.more[key] += [self[key]]
        elif key in self:
            self.more[key] = [self[key]]
        self[key] = datadict
    def _get_derived(self, x, default=None):
        return self._get_base(self.key(x), default)
    def _pop_derived(self, x):
        key = self.key(x)
        res = self[key]
        if key in self.more:
            if len(self.more[key]) == 1:
                self[key] = self.more.pop(key)[0]
            else:
                self[key] = self.more[key].pop(-1)
        return res
class BestSolution(object):
    """container to keep track of the best solution seen"""
    def __init__(self, x=None, f=np.inf, evals=None):
        """initialize the best solution with `x`, `f`, and `evals`.
        Better solutions have smaller `f`-values.

        """
        self.x = x
        self.x_geno = None
        self.f = f if f is not None and f is not np.nan else np.inf
        self.evals = evals
        self.evalsall = evals
        self.last = BlancClass()
        self.last.x = x
        self.last.f = f
    def update(self, arx, xarchive=None, arf=None, evals=None):
        """checks for better solutions in list `arx`, based on the smallest
        corresponding value in `arf`, alternatively, `update` may be called
        with a `BestSolution` instance like ``update(another_best_solution)``
        in which case the better solution becomes the current best.

        `xarchive` is used to retrieve the genotype of a solution.

        """
        if arf is not None:  # find failsave minimum
            minidx = np.nanargmin(arf)
            if minidx is np.nan:
                return
            minarf = arf[minidx]
            # minarf = reduce(lambda x, y: y if y and y is not np.nan and y < x else x, arf, np.inf)
        if type(arx) == BestSolution:
            if self.evalsall is None:
                self.evalsall = arx.evalsall
            elif arx.evalsall is not None:
                self.evalsall = max((self.evalsall, arx.evalsall))
            if arx.f is not None and arx.f < np.inf:
                self.update([arx.x], xarchive, [arx.f], arx.evals)
            return self
        elif minarf < np.inf and (minarf < self.f or self.f is None):
            self.x, self.f = arx[minidx], arf[minidx]
            self.x_geno = xarchive[self.x]['geno'] if xarchive is not None else None
            self.evals = None if not evals else evals - len(arf) + minidx+1
            self.evalsall = evals
        elif evals:
            self.evalsall = evals
        self.last.x = arx[minidx]
        self.last.f = minarf
    def get(self):
        """return ``(x, f, evals)`` """
        return self.x, self.f, self.evals, self.x_geno

#____________________________________________________________
#____________________________________________________________
#
class BoundPenalty(object):
    """Computes the boundary penalty. Must be updated each iteration,
    using the `update` method.

    Details
    -------
    The penalty computes like ``sum(w[i] * (x[i]-xfeas[i])**2)``,
    where `xfeas` is the closest feasible (in-bounds) solution from `x`.
    The weight `w[i]` should be updated during each iteration using
    the update method.

    This class uses `GenoPheno.into_bounds` in method `update` to access
    domain boundary values and repair. This inconsistency is going to be
    removed in future.

    """
    def __init__(self, bounds=None):
        """Argument bounds can be `None` or ``bounds[0]`` and ``bounds[1]``
        are lower  and upper domain boundaries, each is either `None` or
        a scalar or a list or array of appropriate size.
        """
        ##
        # bounds attribute reminds the domain boundary values
        self.bounds = bounds

        self.gamma = 1  # a very crude assumption
        self.weights_initialized = False  # gamma becomes a vector after initialization
        self.hist = []  # delta-f history

    def has_bounds(self):
        """return True, if any variable is bounded"""
        bounds = self.bounds
        if bounds in (None, [None, None]):
            return False
        for i in xrange(bounds[0]):
            if bounds[0][i] is not None and bounds[0][i] > -np.inf:
                return True
        for i in xrange(bounds[1]):
            if bounds[1][i] is not None and bounds[1][i] < np.inf:
                return True
        return False

    def repair(self, x, bounds=None, copy=False, copy_always=False):
        """sets out-of-bounds components of ``x`` on the bounds.

        Arguments
        ---------
            `bounds`
                can be `None`, in which case the "default" bounds are used,
                or ``[lb, ub]``, where `lb` and `ub`
                represent lower and upper domain bounds respectively that
                can be `None` or a scalar or a list or array of length ``len(self)``

        code is more or less copy-paste from Solution.repair, but never tested

        """
        # TODO (old data): CPU(N,lam,iter=20,200,100): 3.3s of 8s for two bounds, 1.8s of 6.5s for one bound
        # TODO: test whether np.max([bounds[0], x], axis=0) etc is speed relevant

        if bounds is None:
            bounds = self.bounds
        if copy_always:
            x_out = array(x, copy=True)
        if bounds not in (None, [None, None], (None, None)):  # solely for effiency
            x_out = array(x, copy=True) if copy and not copy_always else x
            if bounds[0] is not None:
                if np.isscalar(bounds[0]):
                    for i in xrange(len(x)):
                        x_out[i] = max([bounds[0], x[i]])
                else:
                    for i in xrange(len(x)):
                        if bounds[0][i] is not None:
                            x_out[i] = max([bounds[0][i], x[i]])
            if bounds[1] is not None:
                if np.isscalar(bounds[1]):
                    for i in xrange(len(x)):
                        x_out[i] = min([bounds[1], x[i]])
                else:
                    for i in xrange(len(x)):
                        if bounds[1][i] is not None:
                            x_out[i] = min([bounds[1][i], x[i]])
        return x_out  # convenience return

    #____________________________________________________________
    #
    def __call__(self, x, archive, gp):
        """returns the boundary violation penalty for `x` ,where `x` is a
        single solution or a list or array of solutions.
        If `bounds` is not `None`, the values in `bounds` are used, see `__init__`"""
        if x in (None, (), []):
            return x
        if gp.bounds in (None, [None, None], (None, None)):
            return 0.0 if np.isscalar(x[0]) else [0.0] * len(x) # no penalty

        x_is_single_vector = np.isscalar(x[0])
        x = [x] if x_is_single_vector else x

        pen = []
        for xi in x:
            # CAVE: this does not work with already repaired values!!
            # CPU(N,lam,iter=20,200,100)?: 3s of 10s, array(xi): 1s (check again)
            # remark: one deep copy can be prevented by xold = xi first
            xpheno = gp.pheno(archive[xi]['geno'])
            xinbounds = gp.into_bounds(xpheno)
            fac = 1  # exp(0.1 * (log(self.scal) - np.mean(self.scal)))
            pen.append(sum(self.gamma * ((xinbounds - xpheno) / fac)**2) / len(xi))

        return pen[0] if x_is_single_vector else pen

    #____________________________________________________________
    #
    def feasible_ratio(self, solutions):
        """counts for each coordinate the number of feasible values in
        ``solutions`` and returns an array of length ``len(solutions[0])``
        with the ratios.

        `solutions` is a list or array of repaired `Solution` instances

        """
        count = np.zeros(len(solutions[0]))
        for x in solutions:
            count += x.unrepaired == x
        return count / float(len(solutions))

    #____________________________________________________________
    #
    def update(self, function_values, es, bounds=None):
        """updates the weights for computing a boundary penalty.

        Arguments
        ---------
        `function_values`
            all function values of recent population of solutions
        `es`
            `CMAEvolutionStrategy` object instance, in particular the
            method `into_bounds` of the attribute `gp` of type `GenoPheno`
            is used.
        `bounds`
            not (yet) in use other than for ``bounds == [None, None]`` nothing
            is updated.

        Reference: Hansen et al 2009, A Method for Handling Uncertainty...
        IEEE TEC, with addendum at http://www.lri.fr/~hansen/TEC2009online.pdf

        """
        if bounds is None:
            bounds = self.bounds
        if bounds is None or (bounds[0] is None and bounds[1] is None):  # no bounds ==> no penalty
            return self  # len(function_values) * [0.0]  # case without voilations

        N = es.N
        ### prepare
        # compute varis = sigma**2 * C_ii
        varis = es.sigma**2 * array(N * [es.C] if np.isscalar(es.C) else (  # scalar case
                                es.C if np.isscalar(es.C[0]) else  # diagonal matrix case
                                [es.C[i][i] for i in xrange(N)]))  # full matrix case

        # dmean = (es.mean - es.gp.into_bounds(es.mean)) / varis**0.5
        dmean = (es.mean - es.gp.geno(es.gp.into_bounds(es.gp.pheno(es.mean)))) / varis**0.5

        ### Store/update a history of delta fitness value
        fvals = sorted(function_values)
        l = 1 + len(fvals)
        val = fvals[3*l // 4] - fvals[l // 4] # exact interquartile range apart interpolation
        val = val / np.mean(varis)  # new: val is normalized with sigma of the same iteration
        # insert val in history
        if np.isfinite(val) and val > 0:
            self.hist.insert(0, val)
        elif val == inf and len(self.hist) > 1:
            self.hist.insert(0, max(self.hist))
        else:
            pass  # ignore 0 or nan values
        if len(self.hist) > 20 + (3*N) / es.popsize:
            self.hist.pop()

        ### prepare
        dfit = np.median(self.hist)  # median interquartile range
        damp = min(1, es.sp.mueff/10./N)

        ### set/update weights
        # Throw initialization error
        if len(self.hist) == 0:
            raise _Error('wrongful initialization, no feasible solution sampled. ' +
                'Reasons can be mistakenly set bounds (lower bound not smaller than upper bound) or a too large initial sigma0 or... ' +
                'See description of argument func in help(cma.fmin) or an example handling infeasible solutions in help(cma.CMAEvolutionStrategy). ')
        # initialize weights
        if (dmean.any() and (not self.weights_initialized or es.countiter == 2)):  # TODO
            self.gamma = array(N * [2*dfit])
            self.weights_initialized = True
        # update weights gamma
        if self.weights_initialized:
            edist = array(abs(dmean) - 3 * max(1, N**0.5/es.sp.mueff))
            if 1 < 3:  # this is better, around a factor of two
                # increase single weights possibly with a faster rate than they can decrease
                #     value unit of edst is std dev, 3==random walk of 9 steps
                self.gamma *= exp((edist>0) * np.tanh(edist/3) / 2.)**damp
                # decrease all weights up to the same level to avoid single extremely small weights
                #    use a constant factor for pseudo-keeping invariance
                self.gamma[self.gamma > 5 * dfit] *= exp(-1./3)**damp
                #     self.gamma[idx] *= exp(5*dfit/self.gamma[idx] - 1)**(damp/3)
            elif 1 < 3 and (edist>0).any():  # previous method
                # CAVE: min was max in TEC 2009
                self.gamma[edist>0] *= 1.1**min(1, es.sp.mueff/10./N)
                # max fails on cigtab(N=12,bounds=[0.1,None]):
                # self.gamma[edist>0] *= 1.1**max(1, es.sp.mueff/10./N) # this was a bug!?
                # self.gamma *= exp((edist>0) * np.tanh(edist))**min(1, es.sp.mueff/10./N)
            else:  # alternative version, but not better
                solutions = es.pop  # this has not been checked
                r = self.feasible_ratio(solutions)  # has to be the averaged over N iterations
                self.gamma *= exp(np.max([N*[0], 0.3 - r], axis=0))**min(1, es.sp.mueff/10/N)
        es.more_to_write += list(self.gamma) if self.weights_initialized else N * [1.0]
        ### return penalty
        # es.more_to_write = self.gamma if not np.isscalar(self.gamma) else N*[1]
        return self  # bound penalty values

#____________________________________________________________
#____________________________________________________________
#
class GenoPhenoBase(object):
    """depreciated, abstract base class for genotyp-phenotype transformation,
    to be implemented.

    See (and rather use) option ``transformation`` of ``fmin`` or ``CMAEvolutionStrategy``.

    Example
    -------
    ::

        import cma
        class Mygpt(cma.GenoPhenoBase):
            def pheno(self, x):
                return x  # identity for the time being
        gpt = Mygpt()
        optim = cma.CMAEvolutionStrategy(...)
        while not optim.stop():
            X = optim.ask()
            f = [func(gpt.pheno(x)) for x in X]
            optim.tell(X, f)

    In case of a repair, we might pass the repaired solution into `tell()`
    (with check_points being True).

    TODO: check usecases in `CMAEvolutionStrategy` and implement option GenoPhenoBase

    """
    def pheno(self, x):
        raise NotImplementedError()
        return x

#____________________________________________________________
#____________________________________________________________
#
class GenoPheno(object):
    """Genotype-phenotype transformation.

    Method `pheno` provides the transformation from geno- to phenotype,
    that is from the internal representation to the representation used
    in the objective function. Method `geno` provides the "inverse" pheno-
    to genotype transformation. The geno-phenotype transformation comprises,
    in this order:

       - insert fixed variables (with the phenotypic and therefore quite
         possibly "wrong" values)
       - affine linear transformation (scaling and shift)
       - user-defined transformation
       - projection into feasible domain (boundaries)
       - assign fixed variables their original phenotypic value

    By default all transformations are the identity. The boundary
    transformation is only applied, if the boundaries are given as argument to
    the method `pheno` or `geno` respectively.

    ``geno`` is not really necessary and might disappear in future.

    """
    def __init__(self, dim, scaling=None, typical_x=None, bounds=None, fixed_values=None, tf=None):
        """return `GenoPheno` instance with fixed dimension `dim`.

        Keyword Arguments
        -----------------
            `scaling`
                the diagonal of a scaling transformation matrix, multipliers
                in the genotyp-phenotyp transformation, see `typical_x`
            `typical_x`
                ``pheno = scaling*geno + typical_x``
            `bounds` (obsolete, might disappear)
                list with two elements,
                lower and upper bounds both can be a scalar or a "vector"
                of length dim or `None`. Without effect, as `bounds` must
                be given as argument to `pheno()`.
            `fixed_values`
                a dictionary of variable indices and values, like ``{0:2.0, 2:1.1}``,
                that are not subject to change, negative indices are ignored
                (they act like incommenting the index), values are phenotypic
                values.
            `tf`
                list of two user-defined transformation functions, or `None`.

                ``tf[0]`` is a function that transforms the internal representation
                as used by the optimizer into a solution as used by the
                objective function. ``tf[1]`` does the back-transformation.
                For example ::

                    tf_0 = lambda x: [xi**2 for xi in x]
                    tf_1 = lambda x: [abs(xi)**0.5 fox xi in x]

                or "equivalently" without the `lambda` construct ::

                    def tf_0(x):
                        return [xi**2 for xi in x]
                    def tf_1(x):
                        return [abs(xi)**0.5 fox xi in x]

                ``tf=[tf_0, tf_1]`` is a reasonable way to guaranty that only positive
                values are used in the objective function.

        Details
        -------
        If ``tf_1`` is ommitted, the initial x-value must be given as genotype (as the
        phenotype-genotype transformation is unknown) and injection of solutions
        might lead to unexpected results.

        """
        self.N = dim
        self.bounds = bounds
        self.fixed_values = fixed_values
        if tf is not None:
            self.tf_pheno = tf[0]
            self.tf_geno = tf[1]  # TODO: should not necessarily be needed
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r < 1e-7)
            # r = np.random.randn(dim)
            # assert all(tf[0](tf[1](r)) - r > -1e-7)
            print("WARNING in class GenoPheno: user defined transformations have not been tested thoroughly")
        else:
            self.tf_geno = None
            self.tf_pheno = None

        if fixed_values:
            if type(fixed_values) is not dict:
                raise _Error("fixed_values must be a dictionary {index:value,...}")
            if max(fixed_values.keys()) >= dim:
                raise _Error("max(fixed_values.keys()) = " + str(max(fixed_values.keys())) +
                    " >= dim=N=" + str(dim) + " is not a feasible index")
            # convenience commenting functionality: drop negative keys
            for k in list(fixed_values.keys()):
                if k < 0:
                    fixed_values.pop(k)
        if bounds:
            if len(bounds) != 2:
                raise _Error('len(bounds) must be 2 for lower and upper bounds')
            for i in (0,1):
                if bounds[i] is not None:
                    bounds[i] = array(dim * [bounds[i]] if np.isscalar(bounds[i]) else
                                        [b for b in bounds[i]])

        def vec_is_default(vec, default_val=0):
            """return True if `vec` has the value `default_val`,
            None or [None] are also recognized as default"""
            try:
                if len(vec) == 1:
                    vec = vec[0]  # [None] becomes None and is always default
                else:
                    return False
            except TypeError:
                pass  # vec is a scalar

            if vec is None or vec == array(None) or vec == default_val:
                return True
            return False

        self.scales = array(scaling)
        if vec_is_default(self.scales, 1):
            self.scales = 1  # CAVE: 1 is not array(1)
        elif self.scales.shape is not () and len(self.scales) != self.N:
            raise _Error('len(scales) == ' + str(len(self.scales)) +
                         ' does not match dimension N == ' + str(self.N))

        self.typical_x = array(typical_x)
        if vec_is_default(self.typical_x, 0):
            self.typical_x = 0
        elif self.typical_x.shape is not () and len(self.typical_x) != self.N:
            raise _Error('len(typical_x) == ' + str(len(self.typical_x)) +
                         ' does not match dimension N == ' + str(self.N))

        if (self.scales is 1 and
                self.typical_x is 0 and
                self.bounds in (None, [None, None]) and
                self.fixed_values is None and
                self.tf_pheno is None):
            self.isidentity = True
        else:
            self.isidentity = False

    def into_bounds(self, y, bounds=None, copy_never=False, copy_always=False):
        """Argument `y` is a phenotypic vector,
        return `y` put into boundaries, as a copy iff ``y != into_bounds(y)``.

        Note: this code is duplicated in `Solution.repair` and might
        disappear in future.

        """
        bounds = bounds if bounds is not None else self.bounds
        if bounds in (None, [None, None]):
            return y if not copy_always else array(y, copy=True)
        if bounds[0] is not None:
            if len(bounds[0]) not in (1, len(y)):
                raise ValueError('len(bounds[0]) = ' + str(len(bounds[0])) +
                                 ' and len of initial solution (' + str(len(y)) + ') disagree')
            if copy_never:  # is rather slower
                for i in xrange(len(y)):
                    y[i] = max(bounds[0][i], y[i])
            else:
                y = np.max([bounds[0], y], axis=0)
        if bounds[1] is not None:
            if len(bounds[1]) not in (1, len(y)):
                raise ValueError('len(bounds[1]) = ' + str(len(bounds[1])) +
                                    ' and initial solution (' + str(len(y)) + ') disagree')
            if copy_never:
                for i in xrange(len(y)):
                    y[i] = min(bounds[1][i], y[i])
            else:
                y = np.min([bounds[1], y], axis=0)
        return y

    def pheno(self, x, bounds=None, copy=True, copy_always=False):
        """maps the genotypic input argument into the phenotypic space,
        boundaries are only applied if argument ``bounds is not None``, see
        help for class `GenoPheno`

        """
        if copy_always and not copy:
            raise ValueError('arguments copy_always=' + str(copy_always) +
                             ' and copy=' + str(copy) + ' have inconsistent values')
        if self.isidentity and bounds in (None, [None, None], (None, None)):
            return x if not copy_always else array(x, copy=copy_always)

        if self.fixed_values is None:
            y = array(x, copy=copy)  # make a copy, in case
        else:  # expand with fixed values
            y = list(x)  # is a copy
            for i in sorted(self.fixed_values.keys()):
                y.insert(i, self.fixed_values[i])
            y = array(y, copy=False)

        if self.scales is not 1:  # just for efficiency
            y *= self.scales

        if self.typical_x is not 0:
            y += self.typical_x

        if self.tf_pheno is not None:
            y = array(self.tf_pheno(y), copy=False)

        if bounds is not None:
            y = self.into_bounds(y, bounds)

        if self.fixed_values is not None:
            for i, k in list(self.fixed_values.items()):
                y[i] = k

        return y

    def geno(self, y, bounds=None, copy=True, copy_always=False, archive=None):
        """maps the phenotypic input argument into the genotypic space.
        If `bounds` are given, first `y` is projected into the feasible
        domain. In this case ``copy==False`` leads to a copy.

        by default a copy is made only to prevent to modify ``y``

        method geno is only needed if external solutions are injected
        (geno(initial_solution) is depreciated and will disappear)

        TODO: arg copy=True should become copy_never=False

        """
        if archive is not None and bounds is not None:
            try:
                return archive[y]['geno']
            except:
                pass

        x = array(y, copy=(copy and not self.isidentity) or copy_always)

        # bounds = self.bounds if bounds is None else bounds
        if bounds is not None:  # map phenotyp into bounds first
            x = self.into_bounds(x, bounds)

        if self.isidentity:
            return x

        # user-defined transformation
        if self.tf_geno is not None:
            x = array(self.tf_geno(x), copy=False)
        else:
            _Error('t1 of options transformation was not defined but is needed as being the inverse of t0')

        # affine-linear transformation: shift and scaling
        if self.typical_x is not 0:
            x -= self.typical_x
        if self.scales is not 1:  # just for efficiency
            x /= self.scales

        # kick out fixed_values
        if self.fixed_values is not None:
            # keeping the transformed values does not help much
            # therefore it is omitted
            if 1 < 3:
                keys = sorted(self.fixed_values.keys())
                x = array([x[i] for i in range(len(x)) if i not in keys], copy=False)
            else:  # TODO: is this more efficient?
                x = list(x)
                for key in sorted(list(self.fixed_values.keys()), reverse=True):
                    x.remove(key)
                x = array(x, copy=False)
        return x
#____________________________________________________________
#____________________________________________________________
# check out built-in package abc: class ABCMeta, abstractmethod, abstractproperty...
# see http://docs.python.org/whatsnew/2.6.html PEP 3119 abstract base classes
#
class OOOptimizer(object):
    """"abstract" base class for an OO optimizer interface with methods
    `__init__`, `ask`, `tell`, `stop`, `result`, and `optimize`. Only
    `optimize` is fully implemented in this base class.

    Examples
    --------
    All examples minimize the function `elli`, the output is not shown.
    (A preferred environment to execute all examples is ``ipython -pylab``.)
    First we need ::

        from cma import CMAEvolutionStrategy, CMADataLogger  # CMAEvolutionStrategy derives from the OOOptimizer class
        elli = lambda x: sum(1e3**((i-1.)/(len(x)-1.)*x[i])**2 for i in range(len(x)))

    The shortest example uses the inherited method `OOOptimizer.optimize()`::

        res = CMAEvolutionStrategy(8 * [0.1], 0.5).optimize(elli)

    The input parameters to `CMAEvolutionStrategy` are specific to this
    inherited class. The remaining functionality is based on interface
    defined by `OOOptimizer`. We might have a look at the result::

        print(res[0])  # best solution and
        print(res[1])  # its function value

    `res` is the return value from method
    `CMAEvolutionStrategy.result()` appended with `None` (no logger).
    In order to display more exciting output we rather do ::

        logger = CMADataLogger()  # derives from the abstract BaseDataLogger class
        res = CMAEvolutionStrategy(9 * [0.5], 0.3).optimize(elli, logger)
        logger.plot()  # if matplotlib is available, logger == res[-1]

    or even shorter ::

        res = CMAEvolutionStrategy(9 * [0.5], 0.3).optimize(elli, CMADataLogger())
        res[-1].plot()  # if matplotlib is available

    Virtually the same example can be written with an explicit loop
    instead of using `optimize()`. This gives the necessary insight into
    the `OOOptimizer` class interface and gives entire control over the
    iteration loop::

        optim = CMAEvolutionStrategy(9 * [0.5], 0.3)  # a new CMAEvolutionStrategy instance calling CMAEvolutionStrategy.__init__()
        logger = CMADataLogger(optim)  # get a logger instance

        # this loop resembles optimize()
        while not optim.stop(): # iterate
            X = optim.ask()     # get candidate solutions
            f = [elli(x) for x in X]  # evaluate solutions
            #  maybe do something else that needs to be done
            optim.tell(X, f)    # do all the real work: prepare for next iteration
            optim.disp(20)      # display info every 20th iteration
            logger.add()        # log another "data line"

        # final output
        print('termination by', optim.stop())
        print('best f-value =', optim.result()[1])
        print('best solution =', optim.result()[0])
        logger.plot()  # if matplotlib is available
        raw_input('press enter to continue')  # prevents exiting and closing figures

    Details
    -------
    Most of the work is done in the method `tell(...)`. The method `result()` returns
    more useful output.

    """
    def __init__(self, xstart, **more_args):
        """``xstart`` is a mandatory argument"""
        self.xstart = xstart
        self.more_args = more_args
        self.initialize()
    def initialize(self):
        """(re-)set to the initial state"""
        self.countiter = 0
        self.xcurrent = self.xstart[:]
        raise NotImplementedError('method initialize() must be implemented in derived class')
    def ask(self):
        """abstract method, AKA "get" or "sample_distribution", deliver new candidate solution(s), a list of "vectors"
        """
        raise NotImplementedError('method ask() must be implemented in derived class')
    def tell(self, solutions, function_values):
        """abstract method, AKA "update", prepare for next iteration"""
        self.countiter += 1
        raise NotImplementedError('method tell() must be implemented in derived class')
    def stop(self):
        """abstract method, return satisfied termination conditions in a dictionary like
        ``{'termination reason': value, ...}``, for example ``{'tolfun': 1e-12}``, or the empty
        dictionary ``{}``. The implementation of `stop()` should prevent an infinite loop.
        """
        raise NotImplementedError('method stop() is not implemented')
    def disp(self, modulo=None):
        """abstract method, display some iteration infos if ``self.iteration_counter % modulo == 0``"""
        raise NotImplementedError('method disp() is not implemented')
    def result(self):
        """abstract method, return ``(x, f(x), ...)``, that is, the minimizer, its function value, ..."""
        raise NotImplementedError('method result() is not implemented')

    def optimize(self, objectivefct, logger=None, verb_disp=20, iterations=None):
        """find minimizer of `objectivefct` by iterating over `OOOptimizer` `self`
        with verbosity `verb_disp`, using `BaseDataLogger` `logger` with at
        most `iterations` iterations. ::

            return self.result() + (self.stop(), self, logger)

        Example
        -------
        >>> import cma
        >>> res = cma.CMAEvolutionStrategy(7 * [0.1], 0.5).optimize(cma.fcts.rosen, cma.CMADataLogger(), 100)
        (4_w,9)-CMA-ES (mu_w=2.8,w_1=49%) in dimension 7 (seed=630721393)
        Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
            1       9 3.163954777181882e+01 1.0e+00 4.12e-01  4e-01  4e-01 0:0.0
            2      18 3.299006223906629e+01 1.0e+00 3.60e-01  3e-01  4e-01 0:0.0
            3      27 1.389129389866704e+01 1.1e+00 3.18e-01  3e-01  3e-01 0:0.0
          100     900 2.494847340045985e+00 8.6e+00 5.03e-02  2e-02  5e-02 0:0.3
          200    1800 3.428234862999135e-01 1.7e+01 3.77e-02  6e-03  3e-02 0:0.5
          300    2700 3.216640032470860e-04 5.6e+01 6.62e-03  4e-04  9e-03 0:0.8
          400    3600 6.155215286199821e-12 6.6e+01 7.44e-06  1e-07  4e-06 0:1.1
          438    3942 1.187372505161762e-14 6.0e+01 3.27e-07  4e-09  9e-08 0:1.2
          438    3942 1.187372505161762e-14 6.0e+01 3.27e-07  4e-09  9e-08 0:1.2
        ('termination by', {'tolfun': 1e-11})
        ('best f-value =', 1.1189867885201275e-14)
        ('solution =', array([ 1.        ,  1.        ,  1.        ,  0.99999999,  0.99999998,
                0.99999996,  0.99999992]))
        >>> print(res[0])
        [ 1.          1.          1.          0.99999999  0.99999998  0.99999996
          0.99999992]

        """
        if logger is None:
            if hasattr(self, 'logger'):
                logger = self.logger

        citer = 0
        while not self.stop():
            if iterations is not None and citer >= iterations:
                return self.result()
            citer += 1

            X = self.ask()         # deliver candidate solutions
            fitvals = [objectivefct(x) for x in X]
            self.tell(X, fitvals)  # all the work is done here

            self.disp(verb_disp)
            logger.add(self) if logger else None

        logger.add(self, modulo=bool(logger.modulo)) if logger else None
        if verb_disp:
            self.disp(1)
        if verb_disp in (1, True):
            print('termination by', self.stop())
            print('best f-value =', self.result()[1])
            print('solution =', self.result()[0])

        return self.result() + (self.stop(), self, logger)

#____________________________________________________________
#____________________________________________________________
#
class CMAEvolutionStrategy(OOOptimizer):
    """CMA-ES stochastic optimizer class with ask-and-tell interface.

    See `fmin` for the one-line-call functional interface.

    Calling sequence
    ================
    ``optim = CMAEvolutionStrategy(x0, sigma0, opts)``
    returns a class instance.

    Arguments
    ---------
        `x0`
            initial solution, starting point (phenotype).
        `sigma0`
            initial standard deviation.  The problem
            variables should have been scaled, such that a single
            standard deviation on all variables is useful and the
            optimum is expected to lie within about `x0` +- ``3*sigma0``.
            See also options `scaling_of_variables`.
            Often one wants to check for solutions close to the initial
            point. This allows for an easier check for consistency of
            the objective function and its interfacing with the optimizer.
            In this case, a much smaller `sigma0` is advisable.
        `opts`
            options, a dictionary with optional settings,
            see class `Options`.

    Main interface / usage
    ======================
    The ask-and-tell interface is inherited from the generic `OOOptimizer`
    interface for iterative optimization algorithms (see there). With ::

        optim = CMAEvolutionStrategy(8 * [0.5], 0.2)

    an object instance is generated. In each iteration ::

        solutions = optim.ask()

    is used to ask for new candidate solutions (possibly several times) and ::

        optim.tell(solutions, func_values)

    passes the respective function values to `optim`. Instead of `ask()`,
    the class `CMAEvolutionStrategy` also provides ::

        (solutions, func_values) = optim.ask_and_eval(objective_func)

    Therefore, after initialization, an entire optimization can be written
    in two lines like ::

        while not optim.stop():
            optim.tell(*optim.ask_and_eval(objective_func))

    Without the freedom of executing additional lines within the iteration,
    the same reads in a single line as ::

        optim.optimize(objective_func)

    Besides for termination criteria, in CMA-ES only
    the ranks of the `func_values` are relevant.

    Attributes and Properties
    =========================
        - `inputargs` -- passed input arguments
        - `inopts` -- passed options
        - `opts` -- actually used options, some of them can be changed any
          time, see class `Options`
        - `popsize` -- population size lambda, number of candidate solutions
          returned by `ask()`

    Details
    =======
    The following two enhancements are turned off by default.

    **Active CMA** is implemented with option ``CMA_active`` and conducts
    an update of the covariance matrix with negative weights. The
    exponential update is implemented, where from a mathematical
    viewpoint positive definiteness is guarantied. The update is applied
    after the default update and only before the covariance matrix is
    decomposed, which limits the additional computational burden to be
    at most a factor of three (typically smaller). A typical speed up
    factor (number of f-evaluations) is between 1.1 and two.

    References: Jastrebski and Arnold, CEC 2006, Glasmachers et al, GECCO 2010.

    **Selective mirroring** is implemented with option ``CMA_mirrors`` in
    the method ``get_mirror()``. Only the method `ask_and_eval()` will
    then sample selectively mirrored vectors. In selective mirroring, only
    the worst solutions are mirrored. With the default small number of mirrors,
    *pairwise selection* (where at most one of the two mirrors contribute to the
    update of the distribution mean) is implicitely guarantied under selective
    mirroring and therefore not explicitly implemented.

    References: Brockhoff et al, PPSN 2010, Auger et al, GECCO 2011.

    Examples
    ========
    Super-short example, with output shown:

    >>> import cma
    >>> # construct an object instance in 4-D, sigma0=1
    >>> es = cma.CMAEvolutionStrategy(4 * [1], 1, {'seed':234})
    (4_w,8)-CMA-ES (mu_w=2.6,w_1=52%) in dimension 4 (seed=234)
    >>>
    >>> # iterate until termination
    >>> while not es.stop():
    ...    X = es.ask()
    ...    es.tell(X, [cma.fcts.elli(x) for x in X])
    ...    es.disp()  # by default sparse, see option verb_disp
    Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
        1       8 2.093015112685775e+04 1.0e+00 9.27e-01  9e-01  9e-01 0:0.0
        2      16 4.964814235917688e+04 1.1e+00 9.54e-01  9e-01  1e+00 0:0.0
        3      24 2.876682459926845e+05 1.2e+00 1.02e+00  9e-01  1e+00 0:0.0
      100     800 6.809045875281943e-01 1.3e+02 1.41e-02  1e-04  1e-02 0:0.2
      200    1600 2.473662150861846e-10 8.0e+02 3.08e-05  1e-08  8e-06 0:0.5
      233    1864 2.766344961865341e-14 8.6e+02 7.99e-07  8e-11  7e-08 0:0.6
    >>>
    >>> cma.pprint(es.result())
    (Solution([ -1.98546755e-09,  -1.10214235e-09,   6.43822409e-11,
            -1.68621326e-11]),
     4.5119610261406537e-16,
     1666,
     1672,
     209,
     array([ -9.13545269e-09,  -1.45520541e-09,  -6.47755631e-11,
            -1.00643523e-11]),
     array([  3.20258681e-08,   3.15614974e-09,   2.75282215e-10,
             3.27482983e-11]))
    >>>
    >>> # help(es.result) shows
    result(self) method of cma.CMAEvolutionStrategy instance
       return ``(xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xmean), effective_stds)``

    Using the multiprocessing module, we can evaluate the function in parallel with a simple
    modification of the example ::

        import multiprocessing
        # prepare es = ...
        pool = multiprocessing.Pool(es.popsize)
        while not es.stop():
            X = es.ask()
            es.tell(X, pool.map_async(cma.felli, X).get()) # use chunksize parameter as popsize/len(pool)?

    Example with a data logger, lower bounds (at zero) and handling infeasible solutions:

    >>> import cma
    >>> import numpy as np
    >>> es = cma.CMAEvolutionStrategy(10 * [0.2], 0.5, {'bounds': [0, np.inf]})
    >>> logger = cma.CMADataLogger().register(es)
    >>> while not es.stop():
    ...     fit, X = [], []
    ...     while len(X) < es.popsize:
    ...         curr_fit = np.NaN
    ...         while curr_fit is np.NaN:
    ...             x = es.ask(1)[0]
    ...             curr_fit = cma.fcts.somenan(x, cma.fcts.elli) # might return np.NaN
    ...         X.append(x)
    ...         fit.append(curr_fit)
    ...     es.tell(X, fit)
    ...     logger.add()
    ...     es.disp()
    <output omitted>
    >>>
    >>> assert es.result()[1] < 1e-9
    >>> assert es.result()[2] < 9000  # by internal termination
    >>> logger.plot()  # plot data
    >>> cma.show()
    >>> print('  *** if execution stalls close the figure window to continue (and check out ipython --pylab) ***')

    Example implementing restarts with increasing popsize (IPOP), output is not displayed:

    >>> import cma, numpy as np
    >>>
    >>> # restart with increasing population size (IPOP)
    >>> bestever = cma.BestSolution()
    >>> for lam in 10 * 2**np.arange(7):  # 10, 20, 40, 80, ..., 10 * 2**6
    ...     es = cma.CMAEvolutionStrategy('6 - 8 * np.random.rand(9)',  # 9-D
    ...                                   5,         # initial std sigma0
    ...                                   {'popsize': lam,
    ...                                    'verb_append': bestever.evalsall})   # pass options
    ...     logger = cma.CMADataLogger().register(es, append=bestever.evalsall)
    ...     while not es.stop():
    ...         X = es.ask()    # get list of new solutions
    ...         fit = [cma.fcts.rastrigin(x) for x in X]  # evaluate each solution
    ...         es.tell(X, fit) # besides for termination only the ranking in fit is used
    ...
    ...         # display some output
    ...         logger.add()  # add a "data point" to the log, writing in files
    ...         es.disp()  # uses option verb_disp with default 100
    ...
    ...     print('termination:', es.stop())
    ...     cma.pprint(es.best.__dict__)
    ...
    ...     bestever.update(es.best)
    ...
    ...     # show a plot
    ...     logger.plot();
    ...     if bestever.f < 1e-8:  # global optimum was hit
    ...         break
    <output omitted>
    >>> assert es.result()[1] < 1e-8

    On the Rastrigin function, usually after five restarts the global optimum
    is located.

    The final example shows how to resume:

    >>> import cma, pickle
    >>>
    >>> es = cma.CMAEvolutionStrategy(12 * [0.1],  # a new instance, 12-D
    ...                               0.5)         # initial std sigma0
    >>> logger = cma.CMADataLogger().register(es)
    >>> es.optimize(cma.fcts.rosen, logger, iterations=100)
    >>> logger.plot()
    >>> pickle.dump(es, open('saved-cma-object.pkl', 'wb'))
    >>> print('saved')
    >>> del es, logger  # let's start fresh
    >>>
    >>> es = pickle.load(open('saved-cma-object.pkl', 'rb'))
    >>> print('resumed')
    >>> logger = cma.CMADataLogger(es.opts['verb_filenameprefix']  # use same name
    ...                           ).register(es, True)  # True: append to old log data
    >>> es.optimize(cma.fcts.rosen, logger, verb_disp=200)
    >>> assert es.result()[2] < 15000
    >>> cma.pprint(es.result())
    >>> logger.plot()

    Missing Features
    ================
    Option ``randn`` to pass a random number generator.

    :See: `fmin()`, `Options`, `plot()`, `ask()`, `tell()`, `ask_and_eval()`

    """

    # __all__ = ()  # TODO this would be the interface

    #____________________________________________________________
    @property  # read only attribute decorator for a method
    def popsize(self):
        """number of samples by default returned by` ask()`
        """
        return self.sp.popsize

    # this is not compatible with python2.5:
    #     @popsize.setter
    #     def popsize(self, p):
    #         """popsize cannot be set (this might change in future)
    #         """
    #         raise _Error("popsize cannot be changed (this might change in future)")

    #____________________________________________________________
    #____________________________________________________________
    def stop(self, check=True):
        """return a dictionary with the termination status.
        With ``check==False``, the termination conditions are not checked and
        the status might not reflect the current situation.
        """

        if (check and self.countiter > 0 and self.opts['termination_callback'] and
                self.opts['termination_callback'] != str(self.opts['termination_callback'])):
            self.callbackstop = self.opts['termination_callback'](self)

        return self.stopdict(self if check else None)  # update the stopdict and return a Dict

    #____________________________________________________________
    #____________________________________________________________
    def __init__(self, x0, sigma0, inopts = {}):
        """see class `CMAEvolutionStrategy`

        """
        self.inputargs = dict(locals()) # for the record
        del self.inputargs['self'] # otherwise the instance self has a cyclic reference
        self.inopts = inopts
        opts = Options(inopts).complement()  # Options() == fmin([],[]) == defaultOptions()

        if opts['noise_handling'] and eval(opts['noise_handling']):
            raise ValueError('noise_handling not available with class CMAEvolutionStrategy, use function fmin')
        if opts['restarts'] and eval(opts['restarts']):
            raise ValueError('restarts not available with class CMAEvolutionStrategy, use function fmin')

        if x0 == str(x0):
            x0 = eval(x0)
        self.mean = array(x0)  # should not have column or row, is just 1-D
        if self.mean.ndim == 2:
            print('WARNING: input x0 should be a list or 1-D array, trying to flatten ' +
                    str(self.mean.shape) + '-array')
            if self.mean.shape[0] == 1:
                self.mean = self.mean[0]
            elif self.mean.shape[1] == 1:
                self.mean = array([x[0] for x in self.mean])
        if self.mean.ndim != 1:
            raise _Error('x0 must be 1-D array')
        if len(self.mean) <= 1:
            raise _Error('optimization in 1-D is not supported (code was never tested)')

        self.N = self.mean.shape[0]
        N = self.N
        self.mean.resize(N) # 1-D array, not really necessary?!
        self.x0 = self.mean
        self.mean = self.x0.copy()  # goes to initialize

        self.sigma0 = sigma0
        if isinstance(sigma0, str):  # TODO: no real need here (do rather in fmin)
            self.sigma0 = eval(sigma0)  # like '1./N' or 'np.random.rand(1)[0]+1e-2'
        if np.size(self.sigma0) != 1 or np.shape(self.sigma0):
            raise _Error('input argument sigma0 must be (or evaluate to) a scalar')
        self.sigma = self.sigma0  # goes to inialize

        # extract/expand options
        opts.evalall(locals())  # using only N
        self.opts = opts

        self.randn = opts['randn']
        self.gp = GenoPheno(N, opts['scaling_of_variables'], opts['typical_x'],
            opts['bounds'], opts['fixed_variables'], opts['transformation'])
        self.boundPenalty = BoundPenalty(self.gp.bounds)
        s = self.gp.geno(self.mean)
        self.mean = self.gp.geno(self.mean, bounds=self.gp.bounds)
        self.N = len(self.mean)
        N = self.N
        if (self.mean != s).any():
            print('WARNING: initial solution is out of the domain boundaries:')
            print('  x0   = ' + str(self.inputargs['x0']))
            print('  ldom = ' + str(self.gp.bounds[0]))
            print('  udom = ' + str(self.gp.bounds[1]))
        self.fmean = np.NaN             # TODO name should change? prints nan (OK with matlab&octave)
        self.fmean_noise_free = 0.  # for output only

        self.sp = CMAParameters(N, opts)
        self.sp0 = self.sp  # looks useless, as it is not a copy

        # initialization of state variables
        self.countiter = 0
        self.countevals = max((0, opts['verb_append'])) if type(opts['verb_append']) is not bool else 0
        self.ps = np.zeros(N)
        self.pc = np.zeros(N)

        stds = np.ones(N)
        self.sigma_vec = np.ones(N) if np.isfinite(self.sp.dampsvec) else 1
        if np.all(self.opts['CMA_teststds']):  # also 0 would not make sense
            stds = self.opts['CMA_teststds']
            if np.size(stds) != N:
                raise _Error('CMA_teststds option must have dimension = ' + str(N))
        if self.opts['CMA_diagonal']:  # is True or > 0
            # linear time and space complexity
            self.B = array(1) # works fine with np.dot(self.B, anything) and self.B.T
            self.C = stds**2  # TODO: remove this!?
            self.dC = self.C
        else:
            self.B = np.eye(N) # identity(N), do not from matlib import *, as eye is a matrix there
            # prevent equal eigenvals, a hack for np.linalg:
            self.C = np.diag(stds**2 * exp(1e-6*(np.random.rand(N)-0.5)))
            self.dC = np.diag(self.C)
            self.Zneg = np.zeros((N, N))
        self.D = stds

        self.flgtelldone = True
        self.itereigenupdated = self.countiter
        self.noiseS = 0  # noise "signal"
        self.hsiglist = []

        if not opts['seed']:
            np.random.seed()
            six_decimals = (time.time() - 1e6 * (time.time() // 1e6))
            opts['seed'] = 1e5 * np.random.rand() + six_decimals + 1e5 * (time.time() % 1)
        opts['seed'] = int(opts['seed'])
        np.random.seed(opts['seed'])

        self.sent_solutions = SolutionDict()
        self.best = BestSolution()

        out = {}  # TODO: obsolete, replaced by method results()?
        out['best'] = self.best
        # out['hsigcount'] = 0
        out['termination'] = {}
        self.out = out

        self.const = BlancClass()
        self.const.chiN = N**0.5*(1-1./(4.*N)+1./(21.*N**2)) # expectation of norm(randn(N,1))

        # attribute for stopping criteria in function stop
        self.stopdict = CMAStopDict()
        self.callbackstop = 0

        self.fit = BlancClass()
        self.fit.fit = []   # not really necessary
        self.fit.hist = []  # short history of best
        self.fit.histbest = []   # long history of best
        self.fit.histmedian = [] # long history of median

        self.more_to_write = []  #[1, 1, 1, 1]  #  N*[1]  # needed when writing takes place before setting

        # say hello
        if opts['verb_disp'] > 0:
            sweighted = '_w' if self.sp.mu > 1 else ''
            smirr = 'mirr%d' % (self.sp.lam_mirr) if self.sp.lam_mirr else ''
            print('(%d' % (self.sp.mu) + sweighted + ',%d' % (self.sp.popsize) + smirr + ')-CMA-ES' +
                  ' (mu_w=%2.1f,w_1=%d%%)' % (self.sp.mueff, int(100*self.sp.weights[0])) +
                  ' in dimension %d (seed=%d, %s)' % (N, opts['seed'], time.asctime())) # + func.__name__
            if opts['CMA_diagonal'] and self.sp.CMA_on:
                s = ''
                if opts['CMA_diagonal'] is not True:
                    s = ' for '
                    if opts['CMA_diagonal'] < np.inf:
                        s += str(int(opts['CMA_diagonal']))
                    else:
                        s += str(np.floor(opts['CMA_diagonal']))
                    s += ' iterations'
                    s += ' (1/ccov=' + str(round(1./(self.sp.c1+self.sp.cmu))) + ')'
                print('   Covariance matrix is diagonal' + s)

    #____________________________________________________________
    #____________________________________________________________
    def ask(self, number=None, xmean=None, sigma_fac=1):
        """get new candidate solutions, sampled from a multi-variate
        normal distribution and transformed to f-representation
        (phenotype) to be evaluated.

        Arguments
        ---------
            `number`
                number of returned solutions, by default the
                population size ``popsize`` (AKA ``lambda``).
            `xmean`
                distribution mean
            `sigma`
                multiplier for internal sample width (standard
                deviation)

        Return
        ------
        A list of N-dimensional candidate solutions to be evaluated

        Example
        -------
        >>> import cma
        >>> es = cma.CMAEvolutionStrategy([0,0,0,0], 0.3)
        >>> while not es.stop() and es.best.f > 1e-6:  # my_desired_target_f_value
        ...     X = es.ask()  # get list of new solutions
        ...     fit = [cma.fcts.rosen(x) for x in X]  # call function rosen with each solution
        ...     es.tell(X, fit)  # feed values

        :See: `ask_and_eval`, `ask_geno`, `tell`

        """
        pop_geno = self.ask_geno(number, xmean, sigma_fac)


        # N,lambda=20,200: overall CPU 7s vs 5s == 40% overhead, even without bounds!
        #                  new data: 11.5s vs 9.5s == 20%
        # TODO: check here, whether this is necessary?
        # return [self.gp.pheno(x, copy=False, bounds=self.gp.bounds) for x in pop]  # probably fine
        # return [Solution(self.gp.pheno(x, copy=False), copy=False) for x in pop]  # here comes the memory leak, now solved
        # pop_pheno = [Solution(self.gp.pheno(x, copy=False), copy=False).repair(self.gp.bounds) for x in pop_geno]
        pop_pheno = [self.gp.pheno(x, copy=True, bounds=self.gp.bounds) for x in pop_geno]

        if not self.gp.isidentity or use_sent_solutions:  # costs 25% in CPU performance with N,lambda=20,200
            # archive returned solutions, first clean up archive
            if self.countiter % 30/self.popsize**0.5 < 1:
                self.sent_solutions.truncate(0, self.countiter - 1 - 3 * self.N/self.popsize**0.5)
            # insert solutions
            for i in xrange(len(pop_geno)):
                self.sent_solutions[pop_pheno[i]] = {'geno': pop_geno[i],
                                            'pheno': pop_pheno[i],
                                            'iteration': self.countiter}
        return pop_pheno

    #____________________________________________________________
    #____________________________________________________________
    def ask_geno(self, number=None, xmean=None, sigma_fac=1):
        """get new candidate solutions in genotyp, sampled from a
        multi-variate normal distribution.

        Arguments are
            `number`
                number of returned solutions, by default the
                population size `popsize` (AKA lambda).
            `xmean`
                distribution mean
            `sigma_fac`
                multiplier for internal sample width (standard
                deviation)

        `ask_geno` returns a list of N-dimensional candidate solutions
        in genotyp representation and is called by `ask`.

        :See: `ask`, `ask_and_eval`

        """

        if number is None or number < 1:
            number = self.sp.popsize
        if xmean is None:
            xmean = self.mean

        if self.countiter == 0:
            self.tic = time.clock()  # backward compatible
            self.elapsed_time = ElapsedTime()

        if self.opts['CMA_AII']:
            if self.countiter == 0:
                self.aii = AII(self.x0, self.sigma0)
            self.flgtelldone = False
            pop = self.aii.ask(number)
            return pop

        sigma = sigma_fac * self.sigma

        # update parameters for sampling the distribution
        #        fac  0      1      10
        # 150-D cigar:
        #           50749  50464   50787
        # 200-D elli:               == 6.9
        #                  99900   101160
        #                 100995   103275 == 2% loss
        # 100-D elli:               == 6.9
        #                 363052   369325  < 2% loss
        #                 365075   365755

        # update distribution
        if self.sp.CMA_on and (
                (self.opts['updatecovwait'] is None and
                 self.countiter >=
                     self.itereigenupdated + 1./(self.sp.c1+self.sp.cmu)/self.N/10
                 ) or
                (self.opts['updatecovwait'] is not None and
                 self.countiter > self.itereigenupdated + self.opts['updatecovwait']
                 )):
            self.updateBD()

        # sample distribution
        if self.flgtelldone:  # could be done in tell()!?
            self.flgtelldone = False
            self.ary = []

        # each row is a solution
        arz = self.randn((number, self.N))
        if 11 < 3:  # mutate along the principal axes only
            perm = np.random.permutation(self.N) # indices for mutated principal component
            for i in xrange(min((len(arz), self.N))):
                # perm = np.random.permutation(self.N)  # random principal component, should be much worse
                l = sum(arz[i]**2)**0.5
                arz[i] *= 0
                if 11 < 3: # mirrored sampling
                    arz[i][perm[int(i/2)]] = l * (2 * (i % 2) - 1)
                else:
                    arz[i][perm[i % self.N]] = l * np.sign(np.random.rand(1) - 0.5)
        if number == self.sp.popsize:
            self.arz = arz  # is never used
        else:
            pass

        if 11 < 3:  # normalize the length to chiN
            for i in xrange(len(arz)):
                # arz[i] *= exp(self.randn(1)[0] / 8)
                ss = sum(arz[i]**2)**0.5
                arz[i] *= self.const.chiN / ss
            # or to average
            # arz *= 1 * self.const.chiN / np.mean([sum(z**2)**0.5 for z in arz])

        # fac = np.mean(sum(arz**2, 1)**0.5)
        # print fac
        # arz *= self.const.chiN / fac
        self.ary = self.sigma_vec * np.dot(self.B, (self.D * arz).T).T
        pop = xmean + sigma * self.ary
        self.evaluations_per_f_value = 1

        return pop

    def get_mirror(self, x):
        """return ``pheno(self.mean - (geno(x) - self.mean))``.

        TODO: this implementation is yet experimental.

        Selectively mirrored sampling improves to a moderate extend but
        overadditively with active CMA for quite understandable reasons.

        Optimal number of mirrors are suprisingly small: 1,2,3 for maxlam=7,13,20
        however note that 3,6,10 are the respective maximal possible mirrors that
        must be clearly suboptimal.

        """
        try:
            # dx = x.geno - self.mean, repair or boundary handling is not taken into account
            dx = self.sent_solutions[x]['geno'] - self.mean
        except:
            print('WARNING: use of geno is depreciated')
            dx = self.gp.geno(x, copy=True) - self.mean
        dx *= sum(self.randn(self.N)**2)**0.5 / self.mahalanobisNorm(dx)
        x = self.mean - dx
        y = self.gp.pheno(x, bounds=self.gp.bounds)
        if not self.gp.isidentity or use_sent_solutions:  # costs 25% in CPU performance with N,lambda=20,200
            self.sent_solutions[y] = {'geno': x,
                                        'pheno': y,
                                        'iteration': self.countiter}
        return y

    def mirror_penalized(self, f_values, idx):
        """obsolete and subject to removal (TODO),
        return modified f-values such that for each mirror one becomes worst.

        This function is useless when selective mirroring is applied with no
        more than (lambda-mu)/2 solutions.

        Mirrors are leading and trailing values in ``f_values``.

        """
        assert len(f_values) >= 2 * len(idx)
        m = np.max(np.abs(f_values))
        for i in len(idx):
            if f_values[idx[i]] > f_values[-1-i]:
                f_values[idx[i]] += m
            else:
                f_values[-1-i] += m
        return f_values

    def mirror_idx_cov(self, f_values, idx1):  # will most likely be removed
        """obsolete and subject to removal (TODO),
        return indices for negative ("active") update of the covariance matrix
        assuming that ``f_values[idx1[i]]`` and ``f_values[-1-i]`` are
        the corresponding mirrored values

        computes the index of the worse solution sorted by the f-value of the
        better solution.

        TODO: when the actual mirror was rejected, it is better
        to return idx1 instead of idx2.

        Remark: this function might not be necessary at all: if the worst solution
        is the best mirrored, the covariance matrix updates cancel (cave: weights
        and learning rates), which seems what is desirable. If the mirror is bad,
        as strong negative update is made, again what is desirable.
        And the fitness--step-length correlation is in part addressed by
        using flat weights.

        """
        idx2 = np.arange(len(f_values) - 1, len(f_values) - 1 - len(idx1), -1)
        f = []
        for i in xrange(len(idx1)):
            f.append(min((f_values[idx1[i]], f_values[idx2[i]])))
            # idx.append(idx1[i] if f_values[idx1[i]] > f_values[idx2[i]] else idx2[i])
        return idx2[np.argsort(f)][-1::-1]

    #____________________________________________________________
    #____________________________________________________________
    #
    def ask_and_eval(self, func, args=(), number=None, xmean=None, sigma_fac=1,
                     evaluations=1, aggregation=np.median):
        """samples `number` solutions and evaluates them on `func`, where
        each solution `s` is resampled until ``func(s) not in (numpy.NaN, None)``.

        Arguments
        ---------
            `func`
                objective function
            `args`
                additional parameters for `func`
            `number`
                number of solutions to be sampled, by default
                population size ``popsize`` (AKA lambda)
            `xmean`
                mean for sampling the solutions, by default ``self.mean``.
            `sigma_fac`
                multiplier for sampling width, standard deviation, for example
                to get a small perturbation of solution `xmean`
            `evaluations`
                number of evaluations for each sampled solution
            `aggregation`
                function that aggregates `evaluations` values to
                as single value.

        Return
        ------
        ``(X, fit)``, where
            X -- list of solutions
            fit -- list of respective function values

        Details
        -------
        When ``func(x)`` returns `NaN` or `None` a new solution is sampled until
        ``func(x) not in (numpy.NaN, None)``.  The argument to `func` can be
        freely modified within `func`.

        Depending on the ``CMA_mirrors`` option, some solutions are not sampled
        independently but as mirrors of other bad solutions. This is a simple
        derandomization that can save 10-30% of the evaluations in particular
        with small populations, for example on the cigar function.

        Example
        -------
        >>> import cma
        >>> x0, sigma0 = 8*[10], 1  # 8-D
        >>> es = cma.CMAEvolutionStrategy(x0, sigma0)
        >>> while not es.stop():
        ...     X, fit = es.ask_and_eval(cma.fcts.elli)  # handles NaN with resampling
        ...     es.tell(X, fit)  # pass on fitness values
        ...     es.disp(20) # print every 20-th iteration
        >>> print('terminated on ' + str(es.stop()))
        <output omitted>

        A single iteration step can be expressed in one line, such that
        an entire optimization after initialization becomes
        ::

            while not es.stop():
                es.tell(*es.ask_and_eval(cma.fcts.elli))

        """
        # initialize
        popsize = self.sp.popsize
        if number is not None:
            popsize = number
        selective_mirroring = True
        nmirrors = self.sp.lam_mirr
        if popsize != self.sp.popsize:
            nmirrors = Mh.sround(popsize * self.sp.lam_mirr / self.sp.popsize)
            # TODO: now selective mirroring might be impaired
        assert nmirrors <= popsize // 2
        self.mirrors_idx = np.arange(nmirrors)  # might never be used
        self.mirrors_rejected_idx = []  # might never be used
        if xmean is None:
            xmean = self.mean

        # do the work
        fit = []  # or np.NaN * np.empty(number)
        X_first = self.ask(popsize)
        X = []
        for k in xrange(int(popsize)):
            nreject = -1
            f = np.NaN
            while f in (np.NaN, None):  # rejection sampling
                nreject += 1
                if k < popsize - nmirrors or nreject:
                    if nreject:
                        x = self.ask(1, xmean, sigma_fac)[0]
                    else:
                        x = X_first.pop(0)
                else:  # mirrored sample
                    if k == popsize - nmirrors and selective_mirroring:
                        self.mirrors_idx = np.argsort(fit)[-1:-1-nmirrors:-1]
                    x = self.get_mirror(X[self.mirrors_idx[popsize - 1 - k]])
                if nreject == 1 and k >= popsize - nmirrors:
                    self.mirrors_rejected_idx.append(k)

                # contraints handling test hardwired ccccccccccc
                if 11 < 3 and self.opts['vv'] and nreject < 2:  # trying out negative C-update as constraints handling
                    if not hasattr(self, 'constraints_paths'):
                        k = 1
                        self.constraints_paths = [np.zeros(self.N) for _i in xrange(k)]
                    Izero = np.zeros([self.N, self.N])
                    for i in xrange(self.N):
                        if x[i] < 0:
                            Izero[i][i] = 1
                            self.C -= self.opts['vv'] * Izero
                            Izero[i][i] = 0
                    if 1 < 3 and sum([ (9 + i + 1) * x[i] for i in xrange(self.N)]) > 50e3:
                        self.constraints_paths[0] = 0.9 * self.constraints_paths[0] + 0.1 * (x - self.mean) / self.sigma
                        self.C -= (self.opts['vv'] / self.N) * np.outer(self.constraints_paths[0], self.constraints_paths[0])

                f = func(x, *args)
                if f not in (np.NaN, None) and evaluations > 1:
                    f = aggregation([f] + [func(x, *args) for _i in xrange(int(evaluations-1))])
                if nreject + 1 % 1000 == 0:
                    print('  %d solutions rejected (f-value NaN or None) at iteration %d' %
                          (nreject, self.countiter))
            fit.append(f)
            X.append(x)
        self.evaluations_per_f_value = int(evaluations)
        return X, fit


    #____________________________________________________________
    def tell(self, solutions, function_values, check_points=None, copy=False):
        """pass objective function values to prepare for next
        iteration. This core procedure of the CMA-ES algorithm updates
        all state variables, in particular the two evolution paths, the
        distribution mean, the covariance matrix and a step-size.

        Arguments
        ---------
            `solutions`
                list or array of candidate solution points (of
                type `numpy.ndarray`), most presumably before
                delivered by method `ask()` or `ask_and_eval()`.
            `function_values`
                list or array of objective function values
                corresponding to the respective points. Beside for termination
                decisions, only the ranking of values in `function_values`
                is used.
            `check_points`
                If ``check_points is None``, only solutions that are not generated
                by `ask()` are possibly clipped (recommended). ``False`` does not clip
                any solution (not recommended).
                If ``True``, clips solutions that realize long steps (i.e. also
                those that are unlikely to be generated with `ask()`). `check_points`
                can be a list of indices to be checked in solutions.
            `copy`
                ``solutions`` can be modified in this routine, if ``copy is False``

        Details
        -------
        `tell()` updates the parameters of the multivariate
        normal search distribution, namely covariance matrix and
        step-size and updates also the attributes `countiter` and
        `countevals`. To check the points for consistency is quadratic
        in the dimension (like sampling points).

        Bugs
        ----
        The effect of changing the solutions delivered by `ask()` depends on whether
        boundary handling is applied. With boundary handling, modifications are
        disregarded. This is necessary to apply the default boundary handling that
        uses unrepaired solutions but might change in future.

        Example
        -------
        ::

            import cma
            func = cma.fcts.elli  # choose objective function
            es = cma.CMAEvolutionStrategy(cma.np.random.rand(10), 1)
            while not es.stop():
               X = es.ask()
               es.tell(X, [func(x) for x in X])
            es.result()  # where the result can be found

        :See: class `CMAEvolutionStrategy`, `ask()`, `ask_and_eval()`, `fmin()`

        """
    #____________________________________________________________
    # TODO: consider an input argument that flags injected trust-worthy solutions (which means
    #       that they can be treated "absolut" rather than "relative")
        if self.flgtelldone:
            raise _Error('tell should only be called once per iteration')

        lam = len(solutions)
        if lam != array(function_values).shape[0]:
            raise _Error('for each candidate solution '
                        + 'a function value must be provided')
        if lam + self.sp.lam_mirr < 3:
            raise _Error('population size ' + str(lam) + ' is too small when option CMA_mirrors * popsize < 0.5')

        if not np.isscalar(function_values[0]):
            if np.isscalar(function_values[0][0]):
                if self.countiter <= 1:
                    print('WARNING: function values are not a list of scalars (further warnings are suppressed)')
                function_values = [val[0] for val in function_values]
            else:
                raise _Error('objective function values must be a list of scalars')


        ### prepare
        N = self.N
        sp = self.sp
        if 11 < 3 and lam != sp.popsize:  # turned off, because mu should stay constant, still not desastrous
            print('WARNING: population size has changed, recomputing parameters')
            self.sp.set(self.opts, lam)  # not really tested
        if lam < sp.mu:  # rather decrease cmean instead of having mu > lambda//2
            raise _Error('not enough solutions passed to function tell (mu>lambda)')

        self.countiter += 1  # >= 1 now
        self.countevals += sp.popsize * self.evaluations_per_f_value
        self.best.update(solutions, self.sent_solutions, function_values, self.countevals)

        flgseparable = self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']
        if not flgseparable and len(self.C.shape) == 1:  # C was diagonal ie 1-D
            # enter non-separable phase (no easy return from here)
            self.B = np.eye(N) # identity(N)
            self.C = np.diag(self.C)
            idx = np.argsort(self.D)
            self.D = self.D[idx]
            self.B = self.B[:,idx]
            self.Zneg = np.zeros((N, N))

        ### manage fitness
        fit = self.fit  # make short cut

        # CPU for N,lam=20,200: this takes 10s vs 7s
        fit.bndpen = self.boundPenalty.update(function_values, self)(solutions, self.sent_solutions, self.gp)
        # for testing:
        # fit.bndpen = self.boundPenalty.update(function_values, self)([s.unrepaired for s in solutions])
        fit.idx = np.argsort(array(fit.bndpen) + array(function_values))
        fit.fit = array(function_values, copy=False)[fit.idx]

        # update output data TODO: this is obsolete!? However: need communicate current best x-value?
        # old: out['recent_x'] = self.gp.pheno(pop[0])
        self.out['recent_x'] = array(solutions[fit.idx[0]])  # TODO: change in a data structure(?) and use current as identify
        self.out['recent_f'] = fit.fit[0]

        # fitness histories
        fit.hist.insert(0, fit.fit[0])
        # if len(self.fit.histbest) < 120+30*N/sp.popsize or  # does not help, as tablet in the beginning is the critical counter-case
        if ((self.countiter % 5) == 0):  # 20 percent of 1e5 gen.
            fit.histbest.insert(0, fit.fit[0])
            fit.histmedian.insert(0, np.median(fit.fit) if len(fit.fit) < 21
                                    else fit.fit[self.popsize // 2])
        if len(fit.histbest) > 2e4: # 10 + 30*N/sp.popsize:
            fit.histbest.pop()
            fit.histmedian.pop()
        if len(fit.hist) > 10 + 30*N/sp.popsize:
            fit.hist.pop()

        if self.opts['CMA_AII']:
            self.aii.tell(solutions, function_values)
            self.flgtelldone = True
            # for output:
            self.mean = self.aii.mean
            self.dC = self.aii.sigmai**2
            self.sigma = self.aii.sigma
            self.D = 1e-11 + (self.aii.r**2)**0.5
            self.more_to_write += [self.aii.sigma_r]
            return

        # TODO: clean up inconsistency when an unrepaired solution is available and used
        pop = []  # create pop from input argument solutions
        for s in solutions:  # use phenotype before Solution.repair()
            if use_sent_solutions:
                x = self.sent_solutions.pop(s, None)  # 12.7s vs 11.3s with N,lambda=20,200
                if x is not None:
                    pop.append(x['geno'])
                    # TODO: keep additional infos or don't pop s from sent_solutions in the first place
                else:
                    # print 'WARNING: solution not found in ``self.sent_solutions`` (is expected for injected solutions)'
                    pop.append(self.gp.geno(s, copy=copy))  # cannot recover the original genotype with boundary handling
                    if check_points in (None, True, 1):
                        self.repair_genotype(pop[-1])  # necessary if pop[-1] was changed or injected by the user.
            else:  # TODO: to be removed?
                # print 'WARNING: ``geno`` mapping depreciated'
                pop.append(self.gp.geno(s, copy=copy))
                if check_points in (None, True, 1):
                    self.repair_genotype(pop[-1])  # necessary or not?
                # print 'repaired'

        mold = self.mean
        sigma_fac = 1

        # check and normalize each x - m
        # check_points is a flag (None is default: check non-known solutions) or an index list
        # should also a number possible (first check_points points)?
        if check_points not in (None, False, 0, [], ()):  # useful in case of injected solutions and/or adaptive encoding, however is automatic with use_sent_solutions
            try:
                if len(check_points):
                    idx = check_points
            except:
                idx = xrange(sp.popsize)

            for k in idx:
                self.repair_genotype(pop[k])

        # sort pop
        if type(pop) is not array: # only arrays can be multiple indexed
            pop = array(pop, copy=False)

        pop = pop[fit.idx]

        if self.opts['CMA_elitist'] and self.best.f < fit.fit[0]:
            if self.best.x_geno is not None:
                xp = [self.best.x_geno]
                # xp = [self.best.xdict['geno']]
                # xp = [self.gp.geno(self.best.x[:])]  # TODO: remove
                # print self.mahalanobisNorm(xp[0]-self.mean)
                self.clip_or_fit_solutions(xp, [0])
                pop = array([xp[0]] + list(pop))
            else:
                print('genotype for elitist not found')

        # compute new mean
        self.mean = mold + self.sp.cmean * \
                    (sum(sp.weights * pop[0:sp.mu].T, 1) - mold)


        # check Delta m (this is not default, but could become at some point)
        # CAVE: upper_length=sqrt(2)+2 is too restrictive, test upper_length = sqrt(2*N) thoroughly.
        # simple test case injecting self.mean:
        # self.mean = 1e-4 * self.sigma * np.random.randn(N)
        if 11 < 3 and self.opts['vv'] and check_points:  # TODO: check_points might be an index-list
            cmean = self.sp.cmean / min(1, (sqrt(self.opts['vv']*N)+2) / ( # abuse of cmean
                (sqrt(self.sp.mueff) / self.sp.cmean) *
                self.mahalanobisNorm(self.mean - mold)))
        else:
            cmean = self.sp.cmean

        if 11 < 3:  # plot length of mean - mold
            self.more_to_write += [sqrt(sp.mueff) *
                sum(((1./self.D) * dot(self.B.T, self.mean - mold))**2)**0.5 /
                       self.sigma / sqrt(N) / cmean]

        # get learning rate constants
        cc, c1, cmu = sp.cc, sp.c1, sp.cmu
        if flgseparable:
            cc, c1, cmu = sp.cc_sep, sp.c1_sep, sp.cmu_sep

        # now the real work can start

        # evolution paths
        self.ps = (1-sp.cs) * self.ps + \
                  (sqrt(sp.cs*(2-sp.cs)*sp.mueff)  / self.sigma / cmean) * \
                  dot(self.B, (1./self.D) * dot(self.B.T, (self.mean - mold) / self.sigma_vec))

        # "hsig", correction with self.countiter seems not necessary, also pc starts with zero
        hsig = sum(self.ps**2) / (1-(1-sp.cs)**(2*self.countiter)) / self.N < 2 + 4./(N+1)
        if 11 < 3:
            # hsig = 1
            # sp.cc = 4 / (N + 4)
            # sp.cs = 4 / (N + 4)
            # sp.cc = 1
            # sp.damps = 2  #
            # sp.CMA_on = False
            # c1 = 0  # 2 / ((N + 1.3)**2 + 0 * sp.mu) # 1 / N**2
            # cmu = min([1 - c1, cmu])
            if self.countiter == 1:
                print('parameters modified')
        # hsig = sum(self.ps**2) / self.N < 2 + 4./(N+1)
        # adjust missing variance due to hsig, in 4-D with damps=1e99 and sig0 small
        #       hsig leads to premature convergence of C otherwise
        #hsiga = (1-hsig**2) * c1 * cc * (2-cc)  # to be removed in future
        c1a = c1 - (1-hsig**2) * c1 * cc * (2-cc)  # adjust for variance loss

        if 11 < 3:  # diagnostic data
            self.out['hsigcount'] += 1 - hsig
            if not hsig:
                self.hsiglist.append(self.countiter)
        if 11 < 3:  # diagnostic message
            if not hsig:
                print(str(self.countiter) + ': hsig-stall')
        if 11 < 3:  # for testing purpose
            hsig = 1 # TODO:
            #       put correction term, but how?
            if self.countiter == 1:
                print('hsig=1')

        self.pc = (1-cc) * self.pc + \
                  hsig * (sqrt(cc*(2-cc)*sp.mueff) / self.sigma / cmean) * \
                  (self.mean - mold)  / self.sigma_vec

        # covariance matrix adaptation/udpate
        if sp.CMA_on:
            # assert sp.c1 + sp.cmu < sp.mueff / N  # ??
            assert c1 + cmu <= 1

            # default full matrix case
            if not flgseparable:
                Z = (pop[0:sp.mu] - mold) / (self.sigma * self.sigma_vec)
                Z = dot((cmu * sp.weights) * Z.T, Z)  # learning rate integrated
                if self.sp.neg.cmuexp:
                    tmp = (pop[-sp.neg.mu:] - mold) / (self.sigma * self.sigma_vec)
                    self.Zneg *= 1 - self.sp.neg.cmuexp  # for some reason necessary?
                    self.Zneg += dot(sp.neg.weights * tmp.T, tmp) - self.C
                    # self.update_exponential(dot(sp.neg.weights * tmp.T, tmp) - 1 * self.C, -1*self.sp.neg.cmuexp)

                if 11 < 3: # ?3 to 5 times slower??
                    Z = np.zeros((N,N))
                    for k in xrange(sp.mu):
                        z = (pop[k]-mold)
                        Z += np.outer((cmu * sp.weights[k] / (self.sigma * self.sigma_vec)**2) * z, z)

                self.C *= 1 - c1a - cmu
                self.C += np.outer(c1 * self.pc, self.pc) + Z
                self.dC = np.diag(self.C)  # for output and termination checking

            else: # separable/diagonal linear case
                assert(c1+cmu <= 1)
                Z = np.zeros(N)
                for k in xrange(sp.mu):
                    z = (pop[k]-mold) / (self.sigma * self.sigma_vec) # TODO see above
                    Z += sp.weights[k] * z * z  # is 1-D
                self.C = (1-c1a-cmu) * self.C + c1 * self.pc * self.pc + cmu * Z
                # TODO: self.C *= exp(cmuneg * (N - dot(sp.neg.weights,  **2)
                self.dC = self.C
                self.D = sqrt(self.C)  # C is a 1-D array
                self.itereigenupdated = self.countiter

                # idx = self.mirror_idx_cov()  # take half of mirrored vectors for negative update

        # qqqqqqqqqqq
        if 1 < 3 and np.isfinite(sp.dampsvec):
            if self.countiter == 1:
                print("WARNING: CMA_dampsvec option is experimental")
            sp.dampsvec *= np.exp(sp.dampsvec_fading/self.N)
            # TODO: rank-lambda update: *= (1 + sum(z[z>1]**2-1) * exp(sum(z[z<1]**2-1))
            self.sigma_vec *= np.exp((sp.cs/sp.dampsvec/2) * (self.ps**2 - 1))
            # self.sigma_vec *= np.exp((sp.cs/sp.dampsvec) * (abs(self.ps) - (2/np.pi)**0.5))
            self.more_to_write += [exp(np.mean((self.ps**2 - 1)**2))]
            # TODO: rank-mu update

        # step-size adaptation, adapt sigma
        if 1 < 3:  #
            self.sigma *= sigma_fac * \
                            np.exp((min((1, (sp.cs/sp.damps) *
                                    (sqrt(sum(self.ps**2))/self.const.chiN - 1)))))
        else:
            self.sigma *= sigma_fac * \
                            np.exp((min((1000, (sp.cs/sp.damps/2) *
                                    (sum(self.ps**2)/N - 1)))))
        if 11 < 3:
            # derandomized MSR = natural gradient descent using mean(z**2) instead of mu*mean(z)**2
            lengths = array([sum(z**2)**0.5 for z in self.arz[fit.idx[:self.sp.mu]]])
            # print lengths[0::int(self.sp.mu/5)]
            self.sigma *= np.exp(self.sp.mueff**0.5 * dot(self.sp.weights, lengths / self.const.chiN - 1))**(2/(N+1))

        if 11 < 3 and self.opts['vv']:
            if self.countiter < 2:
                print('constant sigma applied')
                print(self.opts['vv'])  # N=10,lam=10: 0.8 is optimal
            self.sigma = self.opts['vv'] * self.sp.mueff * sum(self.mean**2)**0.5 / N

        if self.sigma * min(self.dC)**0.5 < self.opts['minstd']:
            self.sigma = self.opts['minstd'] / min(self.dC)**0.5
        # g = self.countiter
        # N = self.N
        mindx = eval(self.opts['mindx']) if type(self.opts['mindx']) == type('') else self.opts['mindx']
        if self.sigma * min(self.D) < mindx:  # TODO: sigma_vec is missing here
            self.sigma = mindx / min(self.D)

        if self.sigma > 1e9 * self.sigma0:
            alpha = self.sigma / max(self.D)
            self.multiplyC(alpha)
            self.sigma /= alpha**0.5
            self.opts['tolupsigma'] /= alpha**0.5  # to be compared with sigma

        # TODO increase sigma in case of a plateau?

        # Uncertainty noise measurement is done on an upper level

        # output, has moved up, e.g. as part of fmin, TODO to be removed
        if 11 < 3 and self.opts['verb_log'] > 0 and (self.countiter < 4 or
                                          self.countiter % self.opts['verb_log'] == 0):
            # this assumes that two logger with the same name access the same data!
            CMADataLogger(self.opts['verb_filenameprefix']).register(self, append=True).add()
            # self.writeOutput(solutions[fit.idx[0]])

        self.flgtelldone = True
    # end tell()

    def result(self):
        """return ``(xbest, f(xbest), evaluations_xbest, evaluations, iterations, pheno(xmean), effective_stds)``"""
        # TODO: how about xcurrent?
        return self.best.get() + (
            self.countevals, self.countiter, self.gp.pheno(self.mean), self.gp.scales * self.sigma * self.sigma_vec * self.dC**0.5)

    def clip_or_fit_solutions(self, pop, idx):
        """make sure that solutions fit to sample distribution, this interface will probably change.

        In particular the frequency of long vectors appearing in pop[idx] - self.mean is limited.

        """
        for k in idx:
            self.repair_genotype(pop[k])

    def repair_genotype(self, x):
        """make sure that solutions fit to sample distribution, this interface will probably change.

        In particular the frequency of x - self.mean being long is limited.

        """
        mold = self.mean
        if 1 < 3:  # hard clip at upper_length
            upper_length = self.N**0.5 + 2 * self.N / (self.N+2)  # should become an Option, but how? e.g. [0, 2, 2]
            fac = self.mahalanobisNorm(x - mold) / upper_length

            if fac > 1:
                x = (x - mold) / fac + mold
                # print self.countiter, k, fac, self.mahalanobisNorm(pop[k] - mold)
                # adapt also sigma: which are the trust-worthy/injected solutions?
            elif 11 < 3:
                return exp(np.tanh(((upper_length*fac)**2/self.N-1)/2) / 2)
        else:
            if 'checktail' not in self.__dict__:  # hasattr(self, 'checktail')
                raise NotImplementedError
                # from check_tail_smooth import CheckTail  # for the time being
                # self.checktail = CheckTail()
                # print('untested feature checktail is on')
            fac = self.checktail.addchin(self.mahalanobisNorm(x - mold))

            if fac < 1:
                x = fac * (x - mold) + mold

        return 1.0  # sigma_fac, not in use


    #____________________________________________________________
    #____________________________________________________________
    #
    def updateBD(self):
        """update internal variables for sampling the distribution with the
        current covariance matrix C. This method is O(N^3), if C is not diagonal.

        """
        # itereigenupdated is always up-to-date in the diagonal case
        # just double check here
        if self.itereigenupdated == self.countiter:
            return

        if self.sp.neg.cmuexp:  # cave:
            self.update_exponential(self.Zneg, -self.sp.neg.cmuexp)
            # self.C += self.Zpos  # pos update after Zneg would be the correct update, overall:
            # self.C = self.Zpos + Cs * Mh.expms(-self.sp.neg.cmuexp*Csi*self.Zneg*Csi) * Cs
            self.Zneg = np.zeros((self.N, self.N))

        if self.sigma_vec is not 1 and not np.all(self.sigma_vec == 1):
            self.C = dot(dot(np.diag(self.sigma_vec), self.C), np.diag(self.sigma_vec))
            self.sigma_vec[:] = 1

        if self.opts['CMA_const_trace'] in (True, 1, 2):  # normalize trace of C
            if self.opts['CMA_const_trace'] == 2:
                s = np.exp(np.mean(np.log(self.dC)))
            else:
                s = np.mean(self.dC)
            self.C /= s
            self.dC /= s
        self.C = (self.C + self.C.T) / 2
        # self.C = np.triu(self.C) + np.triu(self.C,1).T  # should work as well
        # self.D, self.B = eigh(self.C) # hermitian, ie symmetric C is assumed

        if type(self.opts['CMA_eigenmethod']) == type(1):
            print('WARNING: option CMA_eigenmethod should be a function, not an integer')
            if self.opts['CMA_eigenmethod'] == -1:
                # pygsl
                # easy to install (well, in Windows install gsl binaries first,
                # set system path to respective libgsl-0.dll (or cp the dll to
                # python\DLLS ?), in unzipped pygsl edit
                # gsl_dist/gsl_site_example.py into gsl_dist/gsl_site.py
                # and run "python setup.py build" and "python setup.py install"
                # in MINGW32)
                if 1 < 3:  # import pygsl on the fly
                    try:
                        import pygsl.eigen.eigenvectors  # TODO efficient enough?
                    except ImportError:
                        print('WARNING: could not find pygsl.eigen module, either install pygsl \n' +
                              '  or set option CMA_eigenmethod=1 (is much slower), option set to 1')
                        self.opts['CMA_eigenmethod'] = 0  # use 0 if 1 is too slow

                    self.D, self.B = pygsl.eigen.eigenvectors(self.C)

            elif self.opts['CMA_eigenmethod'] == 0:
                # TODO: thoroughly test np.linalg.eigh
                #       numpy.linalg.eig crashes in 200-D
                #       and EVecs with same EVals are not orthogonal
                self.D, self.B = np.linalg.eigh(self.C)  # self.B[i] is a row and not an eigenvector
            else:  # is overall two;ten times slower in 10;20-D
                self.D, self.B = Misc.eig(self.C)  # def eig, see below
        else:
            self.D, self.B = self.opts['CMA_eigenmethod'](self.C)


        # assert(sum(self.D-DD) < 1e-6)
        # assert(sum(sum(np.dot(BB, BB.T)-np.eye(self.N))) < 1e-6)
        # assert(sum(sum(np.dot(BB * DD, BB.T) - self.C)) < 1e-6)
        idx = np.argsort(self.D)
        self.D = self.D[idx]
        self.B = self.B[:,idx]  # self.B[i] is a row, columns self.B[:,i] are eigenvectors
        # assert(all(self.B[self.countiter % self.N] == self.B[self.countiter % self.N,:]))

        # qqqqqqqqqq
        if 11 < 3:  # limit condition number to 1e13
            climit = 1e13  # cave: conditioncov termination is 1e14
            if self.D[-1] / self.D[0] > climit:
                self.D += self.D[-1] / climit
            for i in xrange(self.N):
                self.C[i][i] += self.D[-1] / climit

        if 11 < 3 and any(abs(sum(self.B[:,0:self.N-1] * self.B[:,1:], 0)) > 1e-6):
            print('B is not orthogonal')
            print(self.D)
            print(sum(self.B[:,0:self.N-1] * self.B[:,1:], 0))
        else:
            # is O(N^3)
            # assert(sum(abs(self.C - np.dot(self.D * self.B,  self.B.T))) < N**2*1e-11)
            pass
        self.D **= 0.5
        self.itereigenupdated = self.countiter

    def multiplyC(self, alpha):
        """multiply C with a scalar and update all related internal variables (dC, D,...)"""
        self.C *= alpha
        if self.dC is not self.C:
            self.dC *= alpha
        self.D *= alpha**0.5
    def update_exponential(self, Z, eta, BDpair=None):
        """exponential update of C that guarantees positive definiteness, that is,
        instead of the assignment ``C = C + eta * Z``,
        C gets C**.5 * exp(eta * C**-.5 * Z * C**-.5) * C**.5.

        Parameter Z should have expectation zero, e.g. sum(w[i] * z[i] * z[i].T) - C
        if E z z.T = C.

        This function conducts two eigendecompositions, assuming that
        B and D are not up to date, unless `BDpair` is given. Given BDpair,
        B is the eigensystem and D is the vector of sqrt(eigenvalues), one
        eigendecomposition is omitted.

        Reference: Glasmachers et al 2010, Exponential Natural Evolution Strategies

        """
        if eta == 0:
            return
        if BDpair:
            B, D = BDpair
        else:
            D, B = self.opts['CMA_eigenmethod'](self.C)
            D **= 0.5
        Csi = dot(B, (B / D).T)
        Cs = dot(B, (B * D).T)
        self.C = dot(Cs, dot(Mh.expms(eta * dot(Csi, dot(Z, Csi)), self.opts['CMA_eigenmethod']), Cs))

    #____________________________________________________________
    #____________________________________________________________
    #
    def _updateCholesky(self, A, Ainv, p, alpha, beta):
        """not yet implemented"""
        # BD is A, p is A*Normal(0,I) distributed
        # input is assumed to be numpy arrays
        # Ainv is needed to compute the evolution path
        # this is a stump and is not tested

        raise _Error("not yet implemented")
        # prepare
        alpha = float(alpha)
        beta = float(beta)
        y = np.dot(Ainv, p)
        y_sum = sum(y**2)

        # compute scalars
        tmp = sqrt(1 + beta * y_sum / alpha)
        fac = (sqrt(alpha) / sum(y**2)) * (tmp - 1)
        facinv = (1. / (sqrt(alpha) * sum(y**2))) * (1 - 1. / tmp)

        # update matrices
        A *= sqrt(alpha)
        A += np.outer(fac * p, y)
        Ainv /= sqrt(alpha)
        Ainv -= np.outer(facinv * y, np.dot(y.T, Ainv))

    #____________________________________________________________
    #____________________________________________________________
    def feedForResume(self, X, function_values):
        """Given all "previous" candidate solutions and their respective
        function values, the state of a `CMAEvolutionStrategy` object
        can be reconstructed from this history. This is the purpose of
        function `feedForResume`.

        Arguments
        ---------
            `X`
              (all) solution points in chronological order, phenotypic
              representation. The number of points must be a multiple
              of popsize.
            `function_values`
              respective objective function values

        Details
        -------
        `feedForResume` can be called repeatedly with only parts of
        the history. The part must have the length of a multiple
        of the population size.
        `feedForResume` feeds the history in popsize-chunks into `tell`.
        The state of the random number generator might not be
        reconstructed, but this would be only relevant for the future.

        Example
        -------
        ::

            import cma

            # prepare
            (x0, sigma0) = ... # initial values from previous trial
            X = ... # list of generated solutions from a previous trial
            f = ... # respective list of f-values

            # resume
            es = cma.CMAEvolutionStrategy(x0, sigma0)
            es.feedForResume(X, f)

            # continue with func as objective function
            while not es.stop():
               X = es.ask()
               es.tell(X, [func(x) for x in X])

        Credits to Dirk Bueche and Fabrice Marchal for the feeding idea.

        :See: class `CMAEvolutionStrategy` for a simple dump/load to resume

        """
        if self.countiter > 0:
            print('WARNING: feed should generally be used with a new object instance')
        if len(X) != len(function_values):
            raise _Error('number of solutions ' + str(len(X)) +
                ' and number function values ' +
                str(len(function_values))+' must not differ')
        popsize = self.sp.popsize
        if (len(X) % popsize) != 0:
            raise _Error('number of solutions ' + str(len(X)) +
                    ' must be a multiple of popsize (lambda) ' +
                    str(popsize))
        for i in xrange(len(X) / popsize):
            # feed in chunks of size popsize
            self.ask()  # a fake ask, mainly for a conditioned calling of updateBD
                        # and secondary to get possibly the same random state
            self.tell(X[i*popsize:(i+1)*popsize], function_values[i*popsize:(i+1)*popsize])

    #____________________________________________________________
    #____________________________________________________________
    def readProperties(self):
        """reads dynamic parameters from property file (not implemented)
        """
        print('not yet implemented')

    #____________________________________________________________
    #____________________________________________________________
    def mahalanobisNorm(self, dx):
        """
        compute the Mahalanobis norm that is induced by the adapted covariance
        matrix C times sigma**2.

        Argument
        --------
        A *genotype* difference `dx`.

        Example
        -------
        >>> import cma, numpy
        >>> es = cma.CMAEvolutionStrategy(numpy.ones(10), 1)
        >>> xx = numpy.random.randn(2, 10)
        >>> d = es.mahalanobisNorm(es.gp.geno(xx[0]-xx[1]))

        `d` is the distance "in" the true sample distribution,
        sampled points have a typical distance of ``sqrt(2*es.N)``,
        where `N` is the dimension. In the example, `d` is the
        Euclidean distance, because C = I and sigma = 1.

        """
        return sqrt(sum((self.D**-1 * np.dot(self.B.T, dx))**2)) / self.sigma

    #____________________________________________________________
    #____________________________________________________________
    #
    def timesCroot(self, mat):
        """return C**0.5 times mat, where mat can be a vector or matrix.
        Not functional, because _Croot=C**0.5 is never computed (should be in updateBD)
        """
        print("WARNING: timesCroot is not yet tested")
        if self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']:
            res = (self._Croot * mat.T).T
        else:
            res = np.dot(self._Croot, mat)
        return res
    def divCroot(self, mat):
        """return C**-1/2 times mat, where mat can be a vector or matrix"""
        print("WARNING: divCroot is not yet tested")
        if self.opts['CMA_diagonal'] is True \
                       or self.countiter <= self.opts['CMA_diagonal']:
            res = (self._Crootinv * mat.T).T
        else:
            res = np.dot(self._Crootinv, mat)
        return res

    #____________________________________________________________
    #____________________________________________________________
    def disp_annotation(self):
        """print annotation for `disp()`"""
        print('Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec')
        sys.stdout.flush()

    #____________________________________________________________
    #____________________________________________________________
    def disp(self, modulo=None):  # TODO: rather assign opt['verb_disp'] as default?
        """prints some infos according to `disp_annotation()`, if
        ``iteration_counter % modulo == 0``

        """
        if modulo is None:
            modulo = self.opts['verb_disp']

        # console display
        if modulo:
            if (self.countiter-1) % (10 * modulo) < 1:
                self.disp_annotation()
            if self.countiter > 0 and (self.stop() or self.countiter < 4
                              or self.countiter % modulo < 1):
                if self.opts['verb_time']:
                    toc = self.elapsed_time()
                    stime = str(int(toc//60))+':'+str(round(toc%60,1))
                else:
                    stime = ''
                print(' '.join((repr(self.countiter).rjust(5),
                                repr(self.countevals).rjust(7),
                                '%.15e' % (min(self.fit.fit)),
                                '%4.1e' % (self.D.max()/self.D.min()),
                                '%6.2e' % self.sigma,
                                '%6.0e' % (self.sigma * sqrt(min(self.dC))),
                                '%6.0e' % (self.sigma * sqrt(max(self.dC))),
                                stime)))
                # if self.countiter < 4:
                sys.stdout.flush()

class Options(dict):
    """``Options()`` returns a dictionary with the available options and their
    default values for function fmin and for class CMAEvolutionStrategy.

    ``Options(opts)`` returns the subset of recognized options in dict(opts).

    ``Options('pop')`` returns a subset of recognized options that contain
    'pop' in there keyword name, value or description.

    Option values can be "written" in a string and, when passed to fmin
    or CMAEvolutionStrategy, are evaluated using "N" and "popsize" as
    known values for dimension and population size (sample size, number
    of new solutions per iteration). All default option values are such
    a string.

    Details
    -------
    All Options are originally defined via the input arguments of
    `fmin()`.

    Options starting with ``tol`` are termination "tolerances".

    For `tolstagnation`, the median over the first and the second half
    of at least `tolstagnation` iterations are compared for both, the
    per-iteration best and per-iteration median function value.
    Some options are, as mentioned (`restarts`,...), only used with `fmin`.

    Example
    -------
    ::

        import cma
        cma.Options('tol')

    is a shortcut for cma.Options().match('tol') that returns all options
    that contain 'tol' in their name or description.

    :See: `fmin`(), `CMAEvolutionStrategy`, `CMAParameters`

    """

    # @classmethod # self is the class, not the instance
    # @property
    # def default(self):
    #     """returns all options with defaults"""
    #     return fmin([],[])

    @staticmethod
    def defaults():
        """return a dictionary with default option values and description,
        calls `fmin([], [])`"""
        return fmin([], [])

    @staticmethod
    def versatileOptions():
        """return list of options that can be changed at any time (not only be
        initialized), however the list might not be entirely up to date. The
        string ' #v ' in the default value indicates a 'versatile' option
        that can be changed any time.

        """
        return tuple(sorted(i[0] for i in list(Options.defaults().items()) if i[1].find(' #v ') > 0))

    def __init__(self, s=None, unchecked=False):
        """return an `Options` instance, either with the default options,
        if ``s is None``, or with all options whose name or description
        contains `s`, if `s` is a string (case is disregarded),
        or with entries from dictionary `s` as options, not complemented
        with default options or settings

        Returns: see above.

        """
        # if not Options.defaults:  # this is different from self.defaults!!!
        #     Options.defaults = fmin([],[])
        if s is None:
            super(Options, self).__init__(Options.defaults())
            # self = Options.defaults()
        elif type(s) is str:
            super(Options, self).__init__(Options().match(s))
            # we could return here
        else:
            super(Options, self).__init__(s)

        if not unchecked:
            for key in list(self.keys()):
                if key not in Options.defaults():
                    print('Warning in cma.Options.__init__(): invalid key ``' + str(key) + '`` popped')
                    self.pop(key)
        # self.evaluated = False  # would become an option entry

    def init(self, dict_or_str, val=None, warn=True):
        """initialize one or several options.

        Arguments
        ---------
            `dict_or_str`
                a dictionary if ``val is None``, otherwise a key.
                If `val` is provided `dict_or_str` must be a valid key.
            `val`
                value for key

        Details
        -------
        Only known keys are accepted. Known keys are in `Options.defaults()`

        """
        #dic = dict_or_key if val is None else {dict_or_key:val}
        dic = dict_or_str
        if val is not None:
            dic = {dict_or_str:val}

        for key, val in list(dic.items()):
            if key not in Options.defaults():
                # TODO: find a better solution?
                if warn:
                    print('Warning in cma.Options.init(): key ' +
                        str(key) + ' ignored')
            else:
                self[key] = val

        return self

    def set(self, dic, val=None, warn=True):
        """set can assign versatile options from `Options.versatileOptions()`
        with a new value, use `init()` for the others.

        Arguments
        ---------
            `dic`
                either a dictionary or a key. In the latter
                case, val must be provided
            `val`
                value for key
            `warn`
                bool, print a warning if the option cannot be changed
                and is therefore omitted

        This method will be most probably used with the ``opts`` attribute of
        a `CMAEvolutionStrategy` instance.

        """
        if val is not None:  # dic is a key in this case
            dic = {dic:val}  # compose a dictionary
        for key, val in list(dic.items()):
            if key in Options.versatileOptions():
                self[key] = val
            elif warn:
                print('Warning in cma.Options.set(): key ' + str(key) + ' ignored')
        return self  # to allow o = Options(o).set(new)

    def complement(self):
        """add all missing options with their default values"""

        for key in Options.defaults():
            if key not in self:
                self[key] = Options.defaults()[key]
        return self

    def settable(self):
        """return the subset of those options that are settable at any
        time.

        Settable options are in `versatileOptions()`, but the
        list might be incomlete.

        """
        return Options([i for i in list(self.items())
                                if i[0] in Options.versatileOptions()])

    def __call__(self, key, default=None, loc=None):
        """evaluate and return the value of option `key` on the fly, or
        returns those options whose name or description contains `key`,
        case disregarded.

        Details
        -------
        Keys that contain `filename` are not evaluated.
        For ``loc==None``, `self` is used as environment
        but this does not define `N`.

        :See: `eval()`, `evalall()`

        """
        try:
            val = self[key]
        except:
            return self.match(key)

        if loc is None:
            loc = self  # TODO: this hack is not so useful: popsize could be there, but N is missing
        try:
            if type(val) is str:
                val = val.split('#')[0].strip()  # remove comments
                if type(val) == type('') and key.find('filename') < 0 and key.find('mindx') < 0:
                    val = eval(val, globals(), loc)
            # invoke default
            # TODO: val in ... fails with array type, because it is applied element wise!
            # elif val in (None,(),[],{}) and default is not None:
            elif val is None and default is not None:
                val = eval(str(default), globals(), loc)
        except:
            pass  # slighly optimistic: the previous is bug-free
        return val

    def eval(self, key, default=None, loc=None):
        """Evaluates and sets the specified option value in
        environment `loc`. Many options need `N` to be defined in
        `loc`, some need `popsize`.

        Details
        -------
        Keys that contain 'filename' are not evaluated.
        For `loc` is None, the self-dict is used as environment

        :See: `evalall()`, `__call__`

        """
        self[key] = self(key, default, loc)
        return self[key]

    def evalall(self, loc=None):
        """Evaluates all option values in environment `loc`.

        :See: `eval()`

        """
        # TODO: this needs rather the parameter N instead of loc
        if 'N' in list(loc.keys()):  # TODO: __init__ of CMA can be simplified
            popsize = self('popsize', Options.defaults()['popsize'], loc)
            for k in list(self.keys()):
                self.eval(k, Options.defaults()[k],
                          {'N':loc['N'], 'popsize':popsize})
        return self

    def match(self, s=''):
        """return all options that match, in the name or the description,
        with string `s`, case is disregarded.

        Example: ``cma.Options().match('verb')`` returns the verbosity options.

        """
        match = s.lower()
        res = {}
        for k in sorted(self):
            s = str(k) + '=\'' + str(self[k]) + '\''
            if match in s.lower():
                res[k] = self[k]
        return Options(res)

    def pp(self):
        pprint(self)

    def printme(self, linebreak=80):
        for i in sorted(Options.defaults().items()):
            s = str(i[0]) + "='" + str(i[1]) + "'"
            a = s.split(' ')

            # print s in chunks
            l = ''  # start entire to the left
            while a:
                while a and len(l) + len(a[0]) < linebreak:
                    l += ' ' + a.pop(0)
                print(l)
                l = '        '  # tab for subsequent lines

#____________________________________________________________
#____________________________________________________________
class CMAParameters(object):
    """strategy parameters like population size and learning rates.

    Note:
        contrary to `Options`, `CMAParameters` is not (yet) part of the
        "user-interface" and subject to future changes (it might become
        a `collections.namedtuple`)

    Example
    -------
    >>> import cma
    >>> es = cma.CMAEvolutionStrategy(20 * [0.1], 1)
    (6_w,12)-CMA-ES (mu_w=3.7,w_1=40%) in dimension 20 (seed=504519190)  # the seed is "random" by default
    >>>
    >>> type(es.sp)  # sp contains the strategy parameters
    <class 'cma.CMAParameters'>
    >>>
    >>> es.sp.disp()
    {'CMA_on': True,
     'N': 20,
     'c1': 0.004181139918745593,
     'c1_sep': 0.034327992810300939,
     'cc': 0.17176721127681213,
     'cc_sep': 0.25259494835857677,
     'cmean': 1.0,
     'cmu': 0.0085149624979034746,
     'cmu_sep': 0.057796356229390715,
     'cs': 0.21434997799189287,
     'damps': 1.2143499779918929,
     'mu': 6,
     'mu_f': 6.0,
     'mueff': 3.7294589343030671,
     'popsize': 12,
     'rankmualpha': 0.3,
     'weights': array([ 0.40240294,  0.25338908,  0.16622156,  0.10437523,  0.05640348,
            0.01720771])}
    >>>
    >> es.sp == cma.CMAParameters(20, 12, cma.Options().evalall({'N': 20}))
    True

    :See: `Options`, `CMAEvolutionStrategy`

    """
    def __init__(self, N, opts, ccovfac=1, verbose=True):
        """Compute strategy parameters, mainly depending on
        dimension and population size, by calling `set`

        """
        self.N = N
        if ccovfac == 1:
            ccovfac = opts['CMA_on']  # that's a hack
        self.set(opts, ccovfac=ccovfac, verbose=verbose)

    def set(self, opts, popsize=None, ccovfac=1, verbose=True):
        """Compute strategy parameters as a function
        of dimension and population size """

        alpha_cc = 1.0  # cc-correction for mueff, was zero before

        def cone(df, mu, N, alphacov=2.0):
            """rank one update learning rate, ``df`` is disregarded and obsolete, reduce alphacov on noisy problems, say to 0.5"""
            return alphacov / ((N + 1.3)**2 + mu)

        def cmu(df, mu, alphamu=0.0, alphacov=2.0):
            """rank mu learning rate, disregarding the constrant cmu <= 1 - cone"""
            c = alphacov * (alphamu + mu - 2 + 1/mu) / ((N + 2)**2 + alphacov * mu / 2)
            # c = alphacov * (alphamu + mu - 2 + 1/mu) / (2 * (N + 2)**1.5 + alphacov * mu / 2)
            # print 'cmu =', c
            return c

        def conedf(df, mu, N):
            """used for computing separable learning rate"""
            return 1. / (df + 2.*sqrt(df) + float(mu)/N)

        def cmudf(df, mu, alphamu):
            """used for computing separable learning rate"""
            return (alphamu + mu - 2. + 1./mu) / (df + 4.*sqrt(df) + mu/2.)

        sp = self
        N = sp.N
        if popsize:
            opts.evalall({'N':N, 'popsize':popsize})
        else:
            popsize = opts.evalall({'N':N})['popsize']  # the default popsize is computed in Options()
        sp.popsize = popsize
        if opts['CMA_mirrors'] < 0.5:
            sp.lam_mirr = int(0.5 + opts['CMA_mirrors'] * popsize)
        elif opts['CMA_mirrors'] > 1:
            sp.lam_mirr = int(0.5 + opts['CMA_mirrors'])
        else:
            sp.lam_mirr = int(0.5 + 0.16 * min((popsize, 2 * N + 2)) + 0.29)  # 0.158650... * popsize is optimal
            # lam = arange(2,22)
            # mirr = 0.16 + 0.29/lam
            # print(lam); print([int(0.5 + l) for l in mirr*lam])
            # [ 2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21]
            # [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4]

        sp.mu_f = sp.popsize / 2.0  # float value of mu
        if opts['CMA_mu'] is not None:
            sp.mu_f = opts['CMA_mu']
        sp.mu = int(sp.mu_f + 0.499999) # round down for x.5
        # in principle we have mu_opt = popsize/2 + lam_mirr/2,
        # which means in particular weights should only be negative for q > 0.5+mirr_frac/2
        if sp.mu > sp.popsize - 2 * sp.lam_mirr + 1:
            print("WARNING: pairwise selection is not implemented, therefore " +
                  " mu = %d > %d = %d - 2*%d + 1 = popsize - 2*mirr + 1 can produce a bias" % (
                    sp.mu, sp.popsize - 2 * sp.lam_mirr + 1, sp.popsize, sp.lam_mirr))
        if sp.lam_mirr > sp.popsize // 2:
            raise _Error("fraction of mirrors in the population as read from option CMA_mirrors cannot be larger 0.5, " +
                         "theoretically optimal is 0.159")
        sp.weights = log(max([sp.mu, sp.popsize / 2.0]) + 0.5) - log(1 + np.arange(sp.mu))
        if 11 < 3:  # equal recombination weights
            sp.mu = sp.popsize // 4
            sp.weights = np.ones(sp.mu)
            print(sp.weights[:10])
        sp.weights /= sum(sp.weights)
        sp.mueff = 1 / sum(sp.weights**2)
        sp.cs = (sp.mueff + 2) / (N + sp.mueff + 3)
        # TODO: clean up (here the cumulation constant is shorter if sigma_vec is used)
        sp.dampsvec = opts['CMA_dampsvec_fac'] * (N + 2) if opts['CMA_dampsvec_fac'] else np.Inf
        sp.dampsvec_fading = opts['CMA_dampsvec_fade']
        if np.isfinite(sp.dampsvec):
            sp.cs = ((sp.mueff + 2) / (N + sp.mueff + 3))**0.5
        # sp.cs = (sp.mueff + 2) / (N + 1.5*sp.mueff + 1)
        sp.cc = (4 + alpha_cc * sp.mueff / N) / (N + 4 + alpha_cc * 2 * sp.mueff / N)
        sp.cc_sep = (1 + 1/N + alpha_cc * sp.mueff / N) / (N**0.5 + 1/N + alpha_cc * 2 * sp.mueff / N) # \not\gg\cc
        sp.rankmualpha = opts['CMA_rankmualpha']
        # sp.rankmualpha = _evalOption(opts['CMA_rankmualpha'], 0.3)
        sp.c1 = ccovfac * min(1, sp.popsize/6) * cone((N**2 + N) / 2, sp.mueff, N) # 2. / ((N+1.3)**2 + sp.mucov)
        sp.c1_sep = ccovfac * conedf(N, sp.mueff, N)
        if 11 < 3:
            sp.c1 = 0.
            print('c1 is zero')
        if opts['CMA_rankmu'] != 0:  # also empty
            sp.cmu = min(1 - sp.c1, ccovfac * cmu((N**2+N)/2, sp.mueff, sp.rankmualpha))
            sp.cmu_sep = min(1 - sp.c1_sep, ccovfac * cmudf(N, sp.mueff, sp.rankmualpha))
        else:
            sp.cmu = sp.cmu_sep = 0

        sp.neg = BlancClass()
        if opts['CMA_active']:
            # in principle we have mu_opt = popsize/2 + lam_mirr/2,
            # which means in particular weights should only be negative for q > 0.5+mirr_frac/2
            sp.neg.mu_f = popsize - (popsize + sp.lam_mirr) / 2  if popsize > 2 else 1
            sp.neg.weights = log(sp.mu_f + 0.5) - log(1 + np.arange(sp.popsize - int(sp.neg.mu_f), sp.popsize))
            sp.neg.mu = len(sp.neg.weights)  # maybe never useful?
            sp.neg.weights /= sum(sp.neg.weights)
            sp.neg.mueff = 1 / sum(sp.neg.weights**2)
            sp.neg.cmuexp = opts['CMA_activefac'] * 0.25 * sp.neg.mueff / ((N+2)**1.5 + 2 * sp.neg.mueff)
            assert sp.neg.mu >= sp.lam_mirr  # not really necessary
            # sp.neg.minresidualvariance = 0.66  # not it use, keep at least 0.66 in all directions, small popsize is most critical
        else:
            sp.neg.cmuexp = 0

        sp.CMA_on = sp.c1 + sp.cmu > 0
        # print(sp.c1_sep / sp.cc_sep)

        if not opts['CMA_on'] and opts['CMA_on'] not in (None,[],(),''):
            sp.CMA_on = False
            # sp.c1 = sp.cmu = sp.c1_sep = sp.cmu_sep = 0

        sp.damps = opts['CMA_dampfac'] * (0.5 +
                                          0.5 * min([1, (sp.lam_mirr/(0.159*sp.popsize) - 1)**2])**1 +
                                          2 * max([0, ((sp.mueff-1) / (N+1))**0.5 - 1]) + sp.cs
                                          )
        if 11 < 3:
            # this is worse than damps = 1 + sp.cs for the (1,10000)-ES on 40D parabolic ridge
            sp.damps = 0.3 + 2 * max([sp.mueff/sp.popsize, ((sp.mueff-1)/(N+1))**0.5 - 1]) + sp.cs
        if 11 < 3:
            # this does not work for lambda = 4*N^2 on the parabolic ridge
            sp.damps = opts['CMA_dampfac'] * (2 - 0*sp.lam_mirr/sp.popsize) * sp.mueff/sp.popsize + 0.3 + sp.cs  # nicer future setting
            print('damps =', sp.damps)
        if 11 < 3:
            sp.damps = 10 * sp.damps  # 1e99 # (1 + 2*max(0,sqrt((sp.mueff-1)/(N+1))-1)) + sp.cs;
            # sp.damps = 20 # 1. + 20 * sp.cs**-1  # 1e99 # (1 + 2*max(0,sqrt((sp.mueff-1)/(N+1))-1)) + sp.cs;
            print('damps is %f' % (sp.damps))

        sp.cmean = float(opts['CMA_cmean'])
        # sp.kappa = 1  # 4-D, lam=16, rank1, kappa < 4 does not influence convergence rate
                        # in larger dim it does, 15-D with defaults, kappa=8 factor 2
        if sp.cmean != 1:
            print('  cmean = %f' % (sp.cmean))

        if verbose:
            if not sp.CMA_on:
                print('covariance matrix adaptation turned off')
            if opts['CMA_mu'] != None:
                print('mu = %f' % (sp.mu_f))

        # return self  # the constructor returns itself

    def disp(self):
        pprint(self.__dict__)

#____________________________________________________________
#____________________________________________________________
class CMAStopDict(dict):
    """keep and update a termination condition dictionary, which is
    "usually" empty and returned by `CMAEvolutionStrategy.stop()`.

    Details
    -------
    This could be a nested class, but nested classes cannot be serialized.

    :See: `stop()`

    """
    def __init__(self, d={}):
        update = (type(d) == CMAEvolutionStrategy)
        inherit = (type(d) == CMAStopDict)
        super(CMAStopDict, self).__init__({} if update else d)
        self._stoplist = d._stoplist if inherit else []    # multiple entries
        self.lastiter = d.lastiter if inherit else 0  # probably not necessary
        if update:
            self._update(d)

    def __call__(self, es):
        """update the dictionary"""
        return self._update(es)

    def _addstop(self, key, cond, val=None):
        if cond:
            self.stoplist.append(key)  # can have the same key twice
            if key in list(self.opts.keys()):
                val = self.opts[key]
            self[key] = val

    def _update(self, es):
        """Test termination criteria and update dictionary.

        """
        if es.countiter == self.lastiter:
            if es.countiter == 0:
                self.__init__()
                return self
            try:
                if es == self.es:
                    return self
            except: # self.es not yet assigned
                pass

        self.lastiter = es.countiter
        self.es = es

        self.stoplist = []

        N = es.N
        opts = es.opts
        self.opts = opts  # a hack to get _addstop going

        # fitness: generic criterion, user defined w/o default
        self._addstop('ftarget',
                     es.best.f < opts['ftarget'])
        # maxiter, maxfevals: generic criteria
        self._addstop('maxfevals',
                     es.countevals - 1 >= opts['maxfevals'])
        self._addstop('maxiter',
                     es.countiter >= opts['maxiter'])
        # tolx, tolfacupx: generic criteria
        # tolfun, tolfunhist (CEC:tolfun includes hist)
        self._addstop('tolx',
                     all([es.sigma*xi < opts['tolx'] for xi in es.pc]) and \
                     all([es.sigma*xi < opts['tolx'] for xi in sqrt(es.dC)]))
        self._addstop('tolfacupx',
                     any([es.sigma * sig > es.sigma0 * opts['tolfacupx']
                          for sig in sqrt(es.dC)]))
        self._addstop('tolfun',
                     es.fit.fit[-1] - es.fit.fit[0] < opts['tolfun'] and \
                     max(es.fit.hist) - min(es.fit.hist) < opts['tolfun'])
        self._addstop('tolfunhist',
                     len(es.fit.hist) > 9 and \
                     max(es.fit.hist) - min(es.fit.hist) <  opts['tolfunhist'])

        # worst seen false positive: table N=80,lam=80, getting worse for fevals=35e3 \approx 50 * N**1.5
        # but the median is not so much getting worse
        # / 5 reflects the sparsity of histbest/median
        # / 2 reflects the left and right part to be compared
        l = int(max(opts['tolstagnation'] / 5. / 2, len(es.fit.histbest) / 10));
        # TODO: why max(..., len(histbest)/10) ???
        # TODO: the problem in the beginning is only with best ==> ???
        if 11 < 3:  #
            print(es.countiter, (opts['tolstagnation'], es.countiter > N * (5 + 100 / es.popsize),
                        len(es.fit.histbest) > 100,
                        np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2*l]),
                        np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2*l])))
        # equality should handle flat fitness
        self._addstop('tolstagnation', # leads sometimes early stop on ftablet, fcigtab, N>=50?
                    1 < 3 and opts['tolstagnation'] and es.countiter > N * (5 + 100 / es.popsize) and
                    len(es.fit.histbest) > 100 and 2*l < len(es.fit.histbest) and
                    np.median(es.fit.histmedian[:l]) >= np.median(es.fit.histmedian[l:2*l]) and
                    np.median(es.fit.histbest[:l]) >= np.median(es.fit.histbest[l:2*l]))
        # iiinteger: stagnation termination can prevent to find the optimum

        self._addstop('tolupsigma', opts['tolupsigma'] and
                      es.sigma / es.sigma0 / np.max(es.D) > opts['tolupsigma'])

        if 11 < 3 and 2*l < len(es.fit.histbest):  # TODO: this might go wrong, because the nb of written columns changes
            tmp = np.array((-np.median(es.fit.histmedian[:l]) + np.median(es.fit.histmedian[l:2*l]),
                        -np.median(es.fit.histbest[:l]) + np.median(es.fit.histbest[l:2*l])))
            es.more_to_write += [(10**t if t < 0 else t + 1) for t in tmp] # the latter to get monotonicy

        if 1 < 3:
            # non-user defined, method specific
            # noeffectaxis (CEC: 0.1sigma), noeffectcoord (CEC:0.2sigma), conditioncov
            self._addstop('noeffectcoord',
                         any([es.mean[i] == es.mean[i] + 0.2*es.sigma*sqrt(es.dC[i])
                              for i in xrange(N)]))
            if opts['CMA_diagonal'] is not True and es.countiter > opts['CMA_diagonal']:
                i = es.countiter % N
                self._addstop('noeffectaxis',
                             sum(es.mean == es.mean + 0.1 * es.sigma * es.D[i] * es.B[:, i]) == N)
            self._addstop('conditioncov',
                         es.D[-1] > 1e7 * es.D[0], 1e14)  # TODO

            self._addstop('callback', es.callbackstop)  # termination_callback
        if len(self):
            self._addstop('flat fitness: please (re)consider how to compute the fitness more elaborate',
                         len(es.fit.hist) > 9 and \
                         max(es.fit.hist) == min(es.fit.hist))
        if 11 < 3 and opts['vv'] == 321:
            self._addstop('||xmean||^2<ftarget', sum(es.mean**2) <= opts['ftarget'])

        return self

#_____________________________________________________________________
#_____________________________________________________________________
#
class BaseDataLogger2(DerivedDictBase):
    """"abstract" base class for a data logger that can be used with an `OOOptimizer`"""
    def add(self, optim=None, more_data=[]):
        """abstract method, add a "data point" from the state of `optim` into the
        logger, the argument `optim` can be omitted if it was `register()`-ed before,
        acts like an event handler"""
        raise NotImplementedError()
    def register(self, optim):
        """abstract method, register an optimizer `optim`, only needed if `add()` is
        called without a value for the `optim` argument"""
        self.optim = optim
    def disp(self):
        """display some data trace (not implemented)"""
        print('method BaseDataLogger.disp() not implemented, to be done in subclass ' + str(type(self)))
    def plot(self):
        """plot data (not implemented)"""
        print('method BaseDataLogger.plot() is not implemented, to be done in subclass ' + str(type(self)))
    def data(self):
        """return logged data in a dictionary (not implemented)"""
        print('method BaseDataLogger.data() is not implemented, to be done in subclass ' + str(type(self)))
class BaseDataLogger(object):
    """"abstract" base class for a data logger that can be used with an `OOOptimizer`"""
    def add(self, optim=None, more_data=[]):
        """abstract method, add a "data point" from the state of `optim` into the
        logger, the argument `optim` can be omitted if it was `register()`-ed before,
        acts like an event handler"""
        raise NotImplementedError()
    def register(self, optim):
        """abstract method, register an optimizer `optim`, only needed if `add()` is
        called without a value for the `optim` argument"""
        self.optim = optim
    def disp(self):
        """display some data trace (not implemented)"""
        print('method BaseDataLogger.disp() not implemented, to be done in subclass ' + str(type(self)))
    def plot(self):
        """plot data (not implemented)"""
        print('method BaseDataLogger.plot() is not implemented, to be done in subclass ' + str(type(self)))
    def data(self):
        """return logged data in a dictionary (not implemented)"""
        print('method BaseDataLogger.data() is not implemented, to be done in subclass ' + str(type(self)))

#_____________________________________________________________________
#_____________________________________________________________________
#
class CMADataLogger(BaseDataLogger):  # might become a dict at some point
    """data logger for class `CMAEvolutionStrategy`. The logger is
    identified by its name prefix and writes or reads according
    data files.

    Examples
    ========
    ::

        import cma
        es = cma.CMAEvolutionStrategy(...)
        data = cma.CMADataLogger().register(es)
        while not es.stop():
            ...
            data.add()  # add can also take an argument

        data.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name


        data2 = cma.CMADataLogger(another_filename_prefix).load()
        data2.plot()
        data2.disp()

    ::

        import cma
        from pylab import *
        res = cma.fmin(cma.Fcts.sphere, rand(10), 1e-0)
        dat = res[-1]  # the CMADataLogger
        dat.load()  # by "default" data are on disk
        semilogy(dat.f[:,0], dat.f[:,5])  # plot f versus iteration, see file header
        show()

    Details
    =======
    After loading data, the logger has the attributes `xmean`, `xrecent`, `std`, `f`, and `D`,
    corresponding to xmean, xrecentbest, stddev, fit, and axlen filename trails.

    :See: `disp()`, `plot()`

    """
    default_prefix = 'outcmaes'
    # names = ('axlen','fit','stddev','xmean','xrecentbest')
    # key_names_with_annotation = ('std', 'xmean', 'xrecent')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy` instance,
        default modulo expands to 1 == log with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[], 'sig':[], 'fit':[], 'xm':[]})
        # class properties:
        self.file_names = ('axlen','fit','stddev','xmean','xrecentbest') # used in load, however hard-coded in add
        self.key_names = ('D', 'f', 'std', 'xmean', 'xrecent') # used in load, however hard-coded in plot
        self.key_names_with_annotation = ('std', 'xmean', 'xrecent') # used in load
        self.modulo = modulo  # allows calling with None
        self.append = append
        self.counter = 0  # number of calls of add, should initial value depend on `append`?
        self.name_prefix = name_prefix if name_prefix else CMADataLogger.default_prefix
        if type(self.name_prefix) == CMAEvolutionStrategy:
            self.name_prefix = self.name_prefix.opts.eval('verb_filenameprefix')
        self.registered = False

    def register(self, es, append=None, modulo=None):
        """register a `CMAEvolutionStrategy` instance for logging,
        ``append=True`` appends to previous data logged under the same name,
        by default previous data are overwritten.

        """
        if type(es) != CMAEvolutionStrategy:
            raise TypeError("only class CMAEvolutionStrategy can be registered for logging")
        self.es = es
        if append is not None:
            self.append = append
        if modulo is not None:
            self.modulo = modulo
        if not self.append and self.modulo != 0:
            self.initialize()  # write file headers
        self.registered = True
        return self

    def initialize(self, modulo=None):
        """reset logger, overwrite original files, `modulo`: log only every modulo call"""
        if modulo is not None:
            self.modulo = modulo
        try:
            es = self.es  # must have been registered
        except AttributeError:
            pass  # TODO: revise usage of es... that this can pass
            raise _Error('call register() before initialize()')

        self.counter = 0  # number of calls of add

        # write headers for output
        fn = self.name_prefix + 'fit.dat'
        strseedtime = 'seed=%d, %s' % (es.opts['seed'], time.asctime())

        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                        'bestever, best, median, worst objective function value, ' +
                        'further objective values of best", ' +
                        strseedtime +
                        # strftime("%Y/%m/%d %H:%M:%S", localtime()) + # just asctime() would do
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'axlen.dat'
        try:
            f = open(fn, 'w')
            f.write('%  columns="iteration, evaluation, sigma, max axis length, ' +
                    ' min axis length, all principle axes lengths ' +
                    ' (sorted square roots of eigenvalues of C)", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()
        fn = self.name_prefix + 'stddev.dat'
        try:
            f = open(fn, 'w')
            f.write('% # columns=["iteration, evaluation, sigma, void, void, ' +
                    ' stds==sigma*sqrt(diag(C))", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()

        fn = self.name_prefix + 'xmean.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, void, void, void, xmean", ' +
                        strseedtime)
                f.write(' # scaling_of_variables: ')
                if np.size(es.gp.scales) > 1:
                    f.write(' '.join(map(str, es.gp.scales)))
                else:
                    f.write(str(es.gp.scales))
                f.write(', typical_x: ')
                if np.size(es.gp.typical_x) > 1:
                    f.write(' '.join(map(str, es.gp.typical_x)))
                else:
                    f.write(str(es.gp.typical_x))
                f.write('\n')
                f.close()
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        fn = self.name_prefix + 'xrecentbest.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # iter+eval+sigma+0+fitness+xbest, ' +
                        strseedtime +
                        '\n')
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        return self
    # end def __init__

    def load(self, filenameprefix=None):
        """loads data from files written and return a data dictionary, *not*
        a prerequisite for using `plot()` or `disp()`.

        Argument `filenameprefix` is the filename prefix of data to be loaded (five files),
        by default ``'outcmaes'``.

        Return data dictionary with keys `xrecent`, `xmean`, `f`, `D`, `std`

        """
        if not filenameprefix:
            filenameprefix = self.name_prefix
        for i in xrange(len(self.file_names)):
            fn = filenameprefix + self.file_names[i] + '.dat'
            try:
                self.__dict__[self.key_names[i]] = _fileToMatrix(fn)
            except:
                print('WARNING: reading from file "' + fn + '" failed')
            if self.key_names[i] in self.key_names_with_annotation:
                self.__dict__[self.key_names[i]].append(self.__dict__[self.key_names[i]][-1])  # copy last row to later fill in annotation position for display
            self.__dict__[self.key_names[i]] = array(self.__dict__[self.key_names[i]], copy=False)
        return self

    def add(self, es=None, more_data=[], modulo=None): # TODO: find a different way to communicate current x and f
        """append some logging data from `CMAEvolutionStrategy` class instance `es`,
        if ``number_of_times_called % modulo`` equals to zero, never if ``modulo==0``.

        The sequence ``more_data`` must always have the same length.

        When used for a different optimizer class, this function can be
        (easily?) adapted by changing the assignments under INTERFACE
        in the implemention.

        """
        self.counter += 1
        mod = modulo if modulo is not None else self.modulo
        if mod == 0 or (self.counter > 3 and self.counter % mod):
            return
        if es is None:
            try:
                es = self.es  # must have been registered
            except AttributeError :
                raise _Error('call `add` with argument `es` or ``register(es)`` before ``add()``')
        elif not self.registered:
            self.register(es) # calls initialize

        # --- INTERFACE, can be changed if necessary ---
        if type(es) is not CMAEvolutionStrategy: # not necessary
            print('WARNING: <type \'CMAEvolutionStrategy\'> expected, found '
                            + str(type(es)) + ' in method CMADataLogger.add')
        evals = es.countevals
        iteration = es.countiter
        sigma = es.sigma
        axratio = es.D.max()/es.D.min()
        xmean = es.mean # TODO: should be optionally phenotype?
        fmean_noise_free = es.fmean_noise_free
        fmean = es.fmean
        try:
            besteverf = es.best.f
            bestf = es.fit.fit[0]
            medianf = es.fit.fit[es.sp.popsize//2]
            worstf = es.fit.fit[-1]
        except:
            if self.counter > 1: # first call without f-values is OK
                raise
        try:
            xrecent = es.best.last.x
        except:
            xrecent = None
        maxD = es.D.max()
        minD = es.D.min()
        diagD = es.D
        diagC = es.sigma*es.sigma_vec*sqrt(es.dC)
        more_to_write = es.more_to_write
        es.more_to_write = []
        # --- end interface ---

        try:

            # fit
            if self.counter > 1:
                fn = self.name_prefix + 'fit.dat'
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + str(axratio) + ' '
                            + str(besteverf) + ' '
                            + '%.16e' % bestf + ' '
                            + str(medianf) + ' '
                            + str(worstf) + ' '
                            # + str(es.sp.popsize) + ' '
                            # + str(10**es.noiseS) + ' '
                            # + str(es.sp.cmean) + ' '
                            + ' '.join(str(i) for i in more_to_write)
                            + ' '.join(str(i) for i in more_data)
                            + '\n')
            # axlen
            fn = self.name_prefix + 'axlen.dat'
            with open(fn, 'a') as f:  # does not rely on reference counting
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(sigma) + ' '
                        + str(maxD) + ' '
                        + str(minD) + ' '
                        + ' '.join(map(str, diagD))
                        + '\n')
            # stddev
            fn = self.name_prefix + 'stddev.dat'
            with open(fn, 'a') as f:
                f.write(str(iteration) + ' '
                        + str(evals) + ' '
                        + str(sigma) + ' '
                        + '0 0 '
                        + ' '.join(map(str, diagC))
                        + '\n')
            # xmean
            fn = self.name_prefix + 'xmean.dat'
            with open(fn, 'a') as f:
                if iteration < 1: # before first iteration
                    f.write('0 0 0 0 0 '
                            + ' '.join(map(str, xmean))
                            + '\n')
                else:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            # + str(sigma) + ' '
                            + '0 '
                            + str(fmean_noise_free) + ' '
                            + str(fmean) + ' '  # TODO: this does not make sense
                            # TODO should be optional the phenotyp?
                            + ' '.join(map(str, xmean))
                            + '\n')
            # xrecent
            fn = self.name_prefix + 'xrecentbest.dat'
            if iteration > 0 and xrecent is not None:
                with open(fn, 'a') as f:
                    f.write(str(iteration) + ' '
                            + str(evals) + ' '
                            + str(sigma) + ' '
                            + '0 '
                            + str(bestf) + ' '
                            + ' '.join(map(str, xrecent))
                            + '\n')

        except (IOError, OSError):
            if iteration <= 1:
                print('could not open/write file')

    def closefig(self):
        pylab.close(self.fighandle)

    def save(self, nameprefix, switch=False):
        """saves logger data to a different set of files, for
        ``switch=True`` also the loggers name prefix is switched to
        the new value

        """
        if not nameprefix or type(nameprefix) is not str:
            _Error('filename prefix must be a nonempty string')

        if nameprefix == self.default_prefix:
            _Error('cannot save to default name "' + nameprefix + '...", chose another name')

        if nameprefix == self.name_prefix:
            return

        for name in CMADataLogger.names:
            open(nameprefix+name+'.dat', 'w').write(open(self.name_prefix+name+'.dat').read())

        if switch:
            self.name_prefix = nameprefix

    def plot(self, fig=None, iabscissa=1, iteridx=None, plot_mean=True,  # TODO: plot_mean default should be False
             foffset=1e-19, x_opt = None, fontsize=10):
        """
        plot data from a `CMADataLogger` (using the files written by the logger).

        Arguments
        ---------
            `fig`
                figure number, by default 325
            `iabscissa`
                ``0==plot`` versus iteration count,
                ``1==plot`` versus function evaluation number
            `iteridx`
                iteration indices to plot

        Return `CMADataLogger` itself.

        Examples
        --------
        ::

            import cma
            logger = cma.CMADataLogger()  # with default name
            # try to plot the "default logging" data (e.g. from previous fmin calls)
            logger.plot() # to continue you might need to close the pop-up window
                          # once and call plot() again.
                          # This behavior seems to disappear in subsequent
                          # calls of plot(). Also using ipython with -pylab
                          # option might help.
            cma.savefig('fig325.png')  # save current figure
            logger.closefig()

        Dependencies: matlabplotlib/pylab.

        """

        dat = self.load(self.name_prefix)

        try:
            # pylab: prodedural interface for matplotlib
            from  matplotlib.pylab import figure, ioff, ion, subplot, semilogy, hold, plot, grid, \
                 axis, title, text, xlabel, isinteractive, draw, gcf

        except ImportError:
            ImportError('could not find matplotlib.pylab module, function plot() is not available')
            return

        if fontsize and pylab.rcParams['font.size'] != fontsize:
            print('global variable pylab.rcParams[\'font.size\'] set (from ' +
                  str(pylab.rcParams['font.size']) + ') to ' + str(fontsize))
            pylab.rcParams['font.size'] = fontsize  # subtracted in the end, but return can happen inbetween

        if fig:
            figure(fig)
        else:
            figure(325)
            # show()  # should not be necessary
        self.fighandle = gcf()  # fighandle.number

        if iabscissa not in (0,1):
            iabscissa = 1
        interactive_status = isinteractive()
        ioff() # prevents immediate drawing

        dat.x = dat.xmean    # this is the genotyp
        if not plot_mean:
            try:
                dat.x = dat.xrecent
            except:
                pass
        if len(dat.x) < 2:
            print('not enough data to plot')
            return {}

        if iteridx is not None:
            dat.f = dat.f[np.where([x in iteridx for x in dat.f[:,0]])[0],:]
            dat.D = dat.D[np.where([x in iteridx for x in dat.D[:,0]])[0],:]
            iteridx.append(dat.x[-1,1])  # last entry is artificial
            dat.x = dat.x[np.where([x in iteridx for x in dat.x[:,0]])[0],:]
            dat.std = dat.std[np.where([x in iteridx for x in dat.std[:,0]])[0],:]

        if iabscissa == 0:
            xlab = 'iterations'
        elif iabscissa == 1:
            xlab = 'function evaluations'

        # use fake last entry in x and std for line extension-annotation
        if dat.x.shape[1] < 100:
            minxend = int(1.06*dat.x[-2, iabscissa])
            # write y-values for individual annotation into dat.x
            dat.x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.x[-2,5:])
            idx2 = np.argsort(idx)
            if x_opt is None:
                dat.x[-1,5+idx] = np.linspace(np.min(dat.x[:,5:]),
                            np.max(dat.x[:,5:]), dat.x.shape[1]-5)
            else:
                dat.x[-1,5+idx] = np.logspace(np.log10(np.min(abs(dat.x[:,5:]))),
                            np.log10(np.max(abs(dat.x[:,5:]))), dat.x.shape[1]-5)
        else:
            minxend = 0

        if len(dat.f) == 0:
            print('nothing to plot')
            return

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        ioff() # turns update off

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where dat.f[:,0]==countiter is monotonous

        subplot(2,2,1)
        self.plotdivers(dat, iabscissa, foffset)

        # TODO: modularize also the remaining subplots
        subplot(2,2,2)
        hold(False)
        if x_opt is not None:  # TODO: differentate neg and pos?
            semilogy(dat.x[:, iabscissa], abs(dat.x[:,5:]) - x_opt, '-')
        else:
            plot(dat.x[:, iabscissa], dat.x[:,5:],'-')
        hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        ax[1] -= 1e-6
        if dat.x.shape[1] < 100:
            yy = np.linspace(ax[2]+1e-6, ax[3]-1e-6, dat.x.shape[1]-5)
            #yyl = np.sort(dat.x[-1,5:])
            idx = np.argsort(dat.x[-1,5:])
            idx2 = np.argsort(idx)
            if x_opt is not None:
                semilogy([dat.x[-1, iabscissa], ax[1]], [abs(dat.x[-1,5:]), yy[idx2]], 'k-') # line from last data point
                semilogy(np.dot(dat.x[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-')
            else:
                # plot([dat.x[-1, iabscissa], ax[1]], [dat.x[-1,5:], yy[idx2]], 'k-') # line from last data point
                plot(np.dot(dat.x[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-')
            # plot(array([dat.x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat.x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat.x[-2,5+idx[i]]))
                text(dat.x[-1,iabscissa], dat.x[-1,5+i], 'x(' + str(i) + ')=' + str(dat.x[-2,5+i]))

        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        title('Object Variables (' + ('mean' if plot_mean else 'curr best') +
                ', ' + str(dat.x.shape[1]-5) + '-D, popsize~' +
                (str(int((dat.f[-1][1] - dat.f[-i][1]) / (dat.f[-1][0] - dat.f[-i][0])))
                    if len(dat.f.T[0]) > 1 and dat.f[-1][0] > dat.f[-i][0] else 'NA')
                + ')')
        # pylab.xticks(xticklocs)

        # Scaling
        subplot(2,2,3)
        hold(False)
        semilogy(dat.D[:, iabscissa], dat.D[:,5:], '-b')
        hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        title('Scaling (All Main Axes)')
        # pylab.xticks(xticklocs)
        xlabel(xlab)

        # standard deviations
        subplot(2,2,4)
        hold(False)
        # remove sigma from stds (graphs become much better readible)
        dat.std[:,5:] = np.transpose(dat.std[:,5:].T / dat.std[:,2].T)
        # ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        # axis(ax)
        if 1 < 2 and dat.std.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06*dat.x[-2, iabscissa])
            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.std[-2,5:])
            idx2 = np.argsort(idx)
            dat.std[-1,5+idx] = np.logspace(np.log10(np.min(dat.std[:,5:])),
                            np.log10(np.max(dat.std[:,5:])), dat.std.shape[1]-5)

            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            yy = np.logspace(np.log10(ax[2]), np.log10(ax[3]), dat.std.shape[1]-5)
            #yyl = np.sort(dat.std[-1,5:])
            idx = np.argsort(dat.std[-1,5:])
            idx2 = np.argsort(idx)
            # plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
            # vertical separator
            plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([np.min(dat.std[-2,5:]), np.max(dat.std[-2,5:])]), 'k-')
            hold(True)
            # plot([dat.std[-1, iabscissa], ax[1]], [dat.std[-1,5:], yy[idx2]], 'k-') # line from last data point
            for i in xrange(len(idx)):
                # text(ax[1], yy[i], ' '+str(idx[i]))
                text(dat.std[-1, iabscissa], dat.std[-1, 5+i], ' '+str(i))
        semilogy(dat.std[:, iabscissa], dat.std[:,5:], '-')
        grid(True)
        title('Standard Deviations in All Coordinates')
        # pylab.xticks(xticklocs)
        xlabel(xlab)
        draw()  # does not suffice
        if interactive_status:
            ion()  # turns interactive mode on (again)
            draw()
        show()

        return self


    #____________________________________________________________
    #____________________________________________________________
    #
    @staticmethod
    def plotdivers(dat, iabscissa, foffset):
        """helper function for `plot()` that plots all what is
        in the upper left subplot like fitness, sigma, etc.

        Arguments
        ---------
            `iabscissa` in ``(0,1)``
                0==versus fevals, 1==versus iteration
            `foffset`
                offset to fitness for log-plot

         :See: `plot()`

        """
        from  matplotlib.pylab import semilogy, hold, grid, \
                 axis, title, text
        fontsize = pylab.rcParams['font.size']

        hold(False)

        dfit = dat.f[:,5]-min(dat.f[:,5])
        dfit[dfit<1e-98] = np.NaN

        if dat.f.shape[1] > 7:
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7]])+foffset,'-k')
            hold(True)

        # (larger indices): additional fitness data, for example constraints values
        if dat.f.shape[1] > 8:
            # dd = abs(dat.f[:,7:]) + 10*foffset
            # dd = np.where(dat.f[:,7:]==0, np.NaN, dd) # cannot be
            semilogy(dat.f[:, iabscissa], np.abs(dat.f[:,8:]) + 10*foffset, 'm')
            hold(True)

        idx = np.where(dat.f[:,5]>1e-98)[0]  # positive values
        semilogy(dat.f[idx, iabscissa], dat.f[idx,5]+foffset, '.b')
        hold(True)
        grid(True)

        idx = np.where(dat.f[:,5] < -1e-98)  # negative values
        semilogy(dat.f[idx, iabscissa], abs(dat.f[idx,5])+foffset,'.r')

        semilogy(dat.f[:, iabscissa],abs(dat.f[:,5])+foffset,'-b')
        semilogy(dat.f[:, iabscissa], dfit, '-c')

        if 11 < 3:  # delta-fitness as points
            dfit = dat.f[1:, 5] - dat.f[:-1,5]  # should be negative usually
            semilogy(dat.f[1:,iabscissa],  # abs(fit(g) - fit(g-1))
                np.abs(dfit)+foffset, '.c')
            i = dfit > 0
            # print(np.sum(i) / float(len(dat.f[1:,iabscissa])))
            semilogy(dat.f[1:,iabscissa][i],  # abs(fit(g) - fit(g-1))
                np.abs(dfit[i])+foffset, '.r')

        # overall minimum
        i = np.argmin(dat.f[:,5])
        semilogy(dat.f[i, iabscissa]*np.ones(2), dat.f[i,5]*np.ones(2), 'rd')
        # semilogy(dat.f[-1, iabscissa]*np.ones(2), dat.f[-1,4]*np.ones(2), 'rd')

        # AR and sigma
        semilogy(dat.f[:, iabscissa], dat.f[:,3], '-r') # AR
        semilogy(dat.f[:, iabscissa], dat.f[:,2],'-g') # sigma
        semilogy(dat.std[:-1, iabscissa], np.vstack([list(map(max, dat.std[:-1,5:])), list(map(min, dat.std[:-1,5:]))]).T,
                     '-m', linewidth=2)
        text(dat.std[-2, iabscissa], max(dat.std[-2, 5:]), 'max std', fontsize=fontsize)
        text(dat.std[-2, iabscissa], min(dat.std[-2, 5:]), 'min std', fontsize=fontsize)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        text(ax[0]+0.01, ax[2], # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
             '.f_recent=' + repr(dat.f[-1,5]) )

        # title('abs(f) (blue), f-min(f) (cyan), Sigma (green), Axis Ratio (red)')
        title('blue:abs(f), cyan:f-min(f), green:sigma, red:axis ratio', fontsize=fontsize-1)
        # pylab.xticks(xticklocs)


    def downsampling(self, factor=10, first=3, switch=True):
        """
        rude downsampling of a `CMADataLogger` data file by `factor`, keeping
        also the first `first` entries. This function is a stump and subject
        to future changes.

        Arguments
        ---------
           - `factor` -- downsampling factor
           - `first` -- keep first `first` entries
           - `switch` -- switch the new logger name to oldname+'down'

        Details
        -------
        ``self.name_prefix+'down'`` files are written

        Example
        -------
        ::

            import cma
            cma.downsampling()  # takes outcmaes* files
            cma.plot('outcmaesdown')

        """
        newprefix = self.name_prefix + 'down'
        for name in CMADataLogger.names:
            f = open(newprefix+name+'.dat','w')
            iline = 0
            cwritten = 0
            for line in open(self.name_prefix+name+'.dat'):
                if iline < first or iline % factor == 0:
                    f.write(line)
                    cwritten += 1
                iline += 1
            f.close()
            print('%d' % (cwritten) + ' lines written in ' + newprefix+name+'.dat')
        if switch:
            self.name_prefix += 'down'
        return self

    #____________________________________________________________
    #____________________________________________________________
    #
    def disp(self, idx=100):  # r_[0:5,1e2:1e9:1e2,-10:0]):
        """displays selected data from (files written by) the class `CMADataLogger`.

        Arguments
        ---------
           `idx`
               indices corresponding to rows in the data file;
               if idx is a scalar (int), the first two, then every idx-th,
               and the last three rows are displayed. Too large index values are removed.

        Example
        -------
        >>> import cma, numpy as np
        >>> res = cma.fmin(cma.fcts.elli, 7 * [0.1], 1, verb_disp=1e9)  # generate data
        >>> assert res[1] < 1e-9
        >>> assert res[2] < 4400
        >>> l = cma.CMADataLogger()  # == res[-1], logger with default name, "points to" above data
        >>> l.disp([0,-1])  # first and last
        >>> l.disp(20)  # some first/last and every 20-th line
        >>> l.disp(np.r_[0:999999:100, -1]) # every 100-th and last
        >>> l.disp(np.r_[0, -10:0]) # first and ten last
        >>> cma.disp(l.name_prefix, np.r_[0::100, -10:])  # the same as l.disp(...)

        Details
        -------
        The data line with the best f-value is displayed as last line.

        :See: `disp()`

        """

        filenameprefix=self.name_prefix

        def printdatarow(dat, iteration):
            """print data of iteration i"""
            i = np.where(dat.f[:, 0] == iteration)[0][0]
            j = np.where(dat.std[:, 0] == iteration)[0][0]
            print('%5d' % (int(dat.f[i,0])) + ' %6d' % (int(dat.f[i,1])) + ' %.14e' % (dat.f[i,5]) +
                  ' %5.1e' % (dat.f[i,3]) +
                  ' %6.2e' % (max(dat.std[j,5:])) + ' %6.2e' % min(dat.std[j,5:]))

        dat = CMADataLogger(filenameprefix).load()
        ndata = dat.f.shape[0]

        # map index to iteration number, is difficult if not all iteration numbers exist
        # idx = idx[np.where(map(lambda x: x in dat.f[:,0], idx))[0]] # TODO: takes pretty long
        # otherwise:
        if idx is None:
            idx = 100
        if np.isscalar(idx):
            # idx = np.arange(0, ndata, idx)
            if idx:
                idx = np.r_[0, 1, idx:ndata-3:idx, -3:0]
            else:
                idx = np.r_[0, 1, -3:0]

        idx = array(idx)
        idx = idx[idx<ndata]
        idx = idx[-idx<=ndata]
        iters = dat.f[idx, 0]
        idxbest = np.argmin(dat.f[:,5])
        iterbest = dat.f[idxbest, 0]

        if len(iters) == 1:
            printdatarow(dat, iters[0])
        else:
            self.disp_header()
            for i in iters:
                printdatarow(dat, i)
            self.disp_header()
            printdatarow(dat, iterbest)
        sys.stdout.flush()
    def disp_header(self):
        heading = 'Iterat Nfevals  function value    axis ratio maxstd   minstd'
        print(heading)

# end class CMADataLogger

#____________________________________________________________
#____________________________________________________________
#
#_____________________________________________________________________
#_____________________________________________________________________
#
class DEAPCMADataLogger(BaseDataLogger):  # might become a dict at some point
    """data logger for class `Strategy`. The logger is
    identified by its name prefix and writes or reads according
    data files.

    Examples
    ========
    ::

        import cma_logger
        es = deap.cma.Strategy(...)
        data = cma_logger.DEAPCMADataLogger().register(es)
        while not es.stop():
            ...
            data.add(fitness_values)  # add can also take `es` as additional argument

        data.plot() # or a short cut can be used:
        cma.plot()  # plot data from logger with default name


        data2 = cma_logger.DEAPCMADataLogger(another_filename_prefix).load()
        data2.plot()
        data2.disp()

    ::

        import cma
        from pylab import *
        res = cma.fmin(cma.Fcts.sphere, rand(10), 1e-0)
        dat = res[-1]  # the CMADataLogger
        dat.load()  # by "default" data are on disk
        semilogy(dat.f[:,0], dat.f[:,5])  # plot f versus iteration, see file header
        show()

    Details
    =======
    After loading data, the logger has the attributes `xmean`, `xrecent`, `std`, `f`, and `D`,
    corresponding to xmean, xrecentbest, stddev, fit, and axlen filename trails.

    :See: `disp()`, `plot()`

    """
    default_prefix = 'outcmaes'
    names = ('axlen','fit','stddev','xmean') # ,'xrecentbest')
    key_names_with_annotation = ('std', 'xmean')

    def __init__(self, name_prefix=default_prefix, modulo=1, append=False):
        """initialize logging of data from a `CMAEvolutionStrategy` instance,
        default modulo expands to 1 == log with each call

        """
        # super(CMAData, self).__init__({'iter':[], 'stds':[], 'D':[], 'sig':[], 'fit':[], 'xm':[]})
        # class properties:
        self.counter = 0  # number of calls of add
        self.best_fitness = np.inf
        self.modulo = modulo  # allows calling with None
        self.append = append
        self.name_prefix = name_prefix if name_prefix else CMADataLogger.default_prefix
        if type(self.name_prefix) == CMAEvolutionStrategy:
            self.name_prefix = self.name_prefix.opts.eval('verb_filenameprefix')
        self.registered = False

    def register(self, es, append=None, modulo=None):
        """register a `CMAEvolutionStrategy` instance for logging,
        ``append=True`` appends to previous data logged under the same name,
        by default previous data are overwritten.

        """
        self.es = es
        if append is not None:
            self.append = append
        if modulo is not None:
            self.modulo = modulo
        if not self.append and self.modulo != 0:
            self.initialize()  # write file headers
        self.registered = True
        return self

    def initialize(self, modulo=None):
        """reset logger, overwrite original files, `modulo`: log only every modulo call"""
        if modulo is not None:
            self.modulo = modulo
        try:
            es = self.es  # must have been registered
        except AttributeError:
            pass  # TODO: revise usage of es... that this can pass
            raise _Error('call register() before initialize()')

        # write headers for output
        fn = self.name_prefix + 'fit.dat'
        if 11 < 3:
            strseedtime = 'seed=%d, %s' % (es.opts['seed'], time.asctime())
        else:
            strseedtime = 'seed=unkown, %s' % (time.asctime())

        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, sigma, axis ratio, ' +
                        'bestever, best, median, worst objective function value, ' +
                        'further objective values of best", ' +
                        strseedtime +
                        # strftime("%Y/%m/%d %H:%M:%S", localtime()) + # just asctime() would do
                        '\n')
        except (IOError, OSError):
            print('could not open file ' + fn)

        fn = self.name_prefix + 'axlen.dat'
        try:
            f = open(fn, 'w')
            f.write('%  columns="iteration, evaluation, sigma, max axis length, ' +
                    ' min axis length, all principle axes lengths ' +
                    ' (sorted square roots of eigenvalues of C)", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()
        fn = self.name_prefix + 'stddev.dat'
        try:
            f = open(fn, 'w')
            f.write('% # columns=["iteration, evaluation, sigma, void, void, ' +
                    ' stds==sigma*sqrt(diag(C))", ' +
                    strseedtime +
                    '\n')
            f.close()
        except (IOError, OSError):
            print('could not open file ' + fn)
        finally:
            f.close()

        fn = self.name_prefix + 'xmean.dat'
        try:
            with open(fn, 'w') as f:
                f.write('% # columns="iteration, evaluation, void, void, void, xmean", ' +
                        strseedtime)
                if 11 < 3:
                    f.write(' # scaling_of_variables: ')
                    if np.size(es.gp.scales) > 1:
                        f.write(' '.join(map(str, es.gp.scales)))
                    else:
                        f.write(str(es.gp.scales))
                    f.write(', typical_x: ')
                    if np.size(es.gp.typical_x) > 1:
                        f.write(' '.join(map(str, es.gp.typical_x)))
                    else:
                        f.write(str(es.gp.typical_x))
                f.write('\n')
                f.close()
        except (IOError, OSError):
            print('could not open/write file ' + fn)

        if 11 < 3:
            fn = self.name_prefix + 'xrecentbest.dat'
            try:
                with open(fn, 'w') as f:
                    f.write('% # iter+eval+sigma+0+fitness+xbest, ' +
                            strseedtime +
                            '\n')
            except (IOError, OSError):
                print('could not open/write file ' + fn)

        return self
    # end def __init__

    def load(self, filenameprefix=None):
        """loads data from files written and return a data dictionary, *not*
        a prerequisite for using `plot()` or `disp()`.

        Argument `filenameprefix` is the filename prefix of data to be loaded (five files),
        by default ``'outcmaes'``.

        Return data dictionary with keys `xrecent`, `xmean`, `f`, `D`, `std`

        """
        if not filenameprefix:
            filenameprefix = self.name_prefix
        dat = self  # historical
        # dat.xrecent = _fileToMatrix(filenameprefix + 'xrecentbest.dat')
        dat.xmean = _fileToMatrix(filenameprefix + 'xmean.dat')
        dat.std = _fileToMatrix(filenameprefix + 'stddev' + '.dat')
        # a hack to later write something into the last entry
        for key in ['xmean', 'std']:  # 'xrecent',
            dat.__dict__[key].append(dat.__dict__[key][-1])  # copy last row to later fill in annotation position for display
            dat.__dict__[key] = array(dat.__dict__[key], copy=False)
        dat.f = array(_fileToMatrix(filenameprefix + 'fit.dat'))
        dat.D = array(_fileToMatrix(filenameprefix + 'axlen' + '.dat'))
        return dat


    def add(self, fitness_values, es=None, more_data=[], modulo=None): # TODO: find a different way to communicate current x and f
        """append some logging data from `CMAEvolutionStrategy` class instance `es`,
        if ``number_of_times_called % modulo`` equals to zero, never if ``modulo==0``.

        The sequence ``more_data`` must always have the same length.

        """
        self.counter += 1
        fitness_values = np.sort(fitness_values)
        if fitness_values[0] < self.best_fitness:
            self.best_fitness = fitness_values[0]
        mod = modulo if modulo is not None else self.modulo
        if mod == 0 or (self.counter > 3 and self.counter % mod):
            return
        if es is None:
            try:
                es = self.es  # must have been registered
            except AttributeError :
                raise _Error('call register() before add() or add(es)')
        elif not self.registered:
            self.register(es)

        if 11 < 3:
            try: # TODO: find a more decent interface to store and pass recent_x
                xrecent = es.best.last.x
            except:
                if self.counter == 2:  # by now a recent_x should be available
                    print('WARNING: es.out[\'recent_x\'] not found in CMADataLogger.add, count='
                          + str(self.counter))
        try:
            # fit
            if es.update_count > 0:
                # fit = es.fit.fit[0]  # TODO: where do we get the fitness from?
                fn = self.name_prefix + 'fit.dat'
                with open(fn, 'a') as f:
                    f.write(str(es.update_count) + ' '
                            + str(es.update_count * es.lambda_) + ' '
                            + str(es.sigma) + ' '
                            + str(es.diagD[-1]/es.diagD[0]) + ' '
                            + str(self.best_fitness) + ' '
                            + '%.16e' % fitness_values[0] + ' '
                            + str(fitness_values[es.lambda_//2]) + ' '
                            + str(fitness_values[-1]) + ' '
                            # + str(es.sp.popsize) + ' '
                            # + str(10**es.noiseS) + ' '
                            # + str(es.sp.cmean) + ' '
                            # + ' '.join(str(i) for i in es.more_to_write)
                            + ' '.join(str(i) for i in more_data)
                            + '\n')
                    # es.more_to_write = []
            # axlen
            fn = self.name_prefix + 'axlen.dat'
            with open(fn, 'a') as f:  # does not rely on reference counting
                f.write(str(es.update_count) + ' '
                        + str(es.update_count * es.lambda_) + ' '
                        + str(es.sigma) + ' '
                        + str(es.diagD[-1]) + ' '
                        + str(es.diagD[0]) + ' '
                        + ' '.join(map(str, es.diagD))
                        + '\n')
            # stddev
            fn = self.name_prefix + 'stddev.dat'
            with open(fn, 'a') as f:
                f.write(str(es.update_count) + ' '
                        + str(es.update_count * es.lambda_) + ' '
                        + str(es.sigma) + ' '
                        + '0 0 '
                        + ' '.join(map(str, es.sigma*np.sqrt([es.C[i][i] for i in xrange(es.dim)])))
                        + '\n')
            # xmean
            fn = self.name_prefix + 'xmean.dat'
            with open(fn, 'a') as f:
                if es.update_count < 1:
                    f.write('0 0 0 0 0 '
                            + ' '.join(map(str,
                                              # TODO should be optional the phenotyp?
                                              # es.gp.geno(es.x0)
                                              es.mean))
                            + '\n')
                else:
                    f.write(str(es.update_count) + ' '
                            + str(es.update_count * es.lambda_) + ' '
                            # + str(es.sigma) + ' '
                            + '0 0 0 '
                            # + str(es.fmean_noise_free) + ' '
                            # + str(es.fmean) + ' '  # TODO: this does not make sense
                            # TODO should be optional the phenotyp?
                            + ' '.join(map(str, es.centroid))
                            + '\n')
            # xrecent
            if 11 < 3:
                fn = self.name_prefix + 'xrecentbest.dat'
                if es.countiter > 0 and xrecent is not None:
                    with open(fn, 'a') as f:
                        f.write(str(es.countiter) + ' '
                                + str(es.countevals) + ' '
                                + str(es.sigma) + ' '
                                + '0 '
                                + str(es.fit.fit[0]) + ' '
                                + ' '.join(map(str, xrecent))
                                + '\n')

        except (IOError, OSError):
            if es.countiter == 1:
                print('could not open/write file')

    def closefig(self):
        pylab.close(self.fighandle)

    def save(self, nameprefix, switch=False):
        """saves logger data to a different set of files, for
        ``switch=True`` also the loggers name prefix is switched to
        the new value

        """
        if not nameprefix or type(nameprefix) is not str:
            _Error('filename prefix must be a nonempty string')

        if nameprefix == self.default_prefix:
            _Error('cannot save to default name "' + nameprefix + '...", chose another name')

        if nameprefix == self.name_prefix:
            return

        for name in CMADataLogger.names:
            open(nameprefix+name+'.dat', 'w').write(open(self.name_prefix+name+'.dat').read())

        if switch:
            self.name_prefix = nameprefix

    def plot(self, fig=None, iabscissa=1, iteridx=None, plot_mean=True,  # TODO: plot_mean default should be False
             foffset=1e-19, x_opt = None, fontsize=10):
        """
        plot data from a `CMADataLogger` (using the files written by the logger).

        Arguments
        ---------
            `fig`
                figure number, by default 325
            `iabscissa`
                ``0==plot`` versus iteration count,
                ``1==plot`` versus function evaluation number
            `iteridx`
                iteration indices to plot

        Return `CMADataLogger` itself.

        Examples
        --------
        ::

            import cma
            logger = cma.CMADataLogger()  # with default name
            # try to plot the "default logging" data (e.g. from previous fmin calls)
            logger.plot() # to continue you might need to close the pop-up window
                          # once and call plot() again.
                          # This behavior seems to disappear in subsequent
                          # calls of plot(). Also using ipython with -pylab
                          # option might help.
            cma.savefig('fig325.png')  # save current figure
            logger.closefig()

        Dependencies: matlabplotlib/pylab.

        """

        dat = self.load(self.name_prefix)

        try:
            # pylab: prodedural interface for matplotlib
            from  matplotlib.pylab import figure, ioff, ion, subplot, semilogy, hold, plot, grid, \
                 axis, title, text, xlabel, isinteractive, draw, gcf

        except ImportError:
            ImportError('could not find matplotlib.pylab module, function plot() is not available')
            return

        if fontsize and pylab.rcParams['font.size'] != fontsize:
            print('global variable pylab.rcParams[\'font.size\'] set (from ' +
                  str(pylab.rcParams['font.size']) + ') to ' + str(fontsize))
            pylab.rcParams['font.size'] = fontsize  # subtracted in the end, but return can happen inbetween

        if fig:
            figure(fig)
        else:
            figure(325)
            # show()  # should not be necessary
        self.fighandle = gcf()  # fighandle.number

        if iabscissa not in (0,1):
            iabscissa = 1
        interactive_status = isinteractive()
        ioff() # prevents immediate drawing

        if 11 < 3:
            dat.x = dat.xrecent
            if len(dat.x) < 2:
                print('not enough data to plot')
                return {}
        # if plot_mean:
        dat.x = dat.xmean    # this is the genotyp
        if iteridx is not None:
            dat.f = dat.f[np.where([x in iteridx for x in dat.f[:,0]])[0],:]
            dat.D = dat.D[np.where([x in iteridx for x in dat.D[:,0]])[0],:]
            iteridx.append(dat.x[-1,1])  # last entry is artificial
            dat.x = dat.x[np.where([x in iteridx for x in dat.x[:,0]])[0],:]
            dat.std = dat.std[np.where([x in iteridx for x in dat.std[:,0]])[0],:]

        if iabscissa == 0:
            xlab = 'iterations'
        elif iabscissa == 1:
            xlab = 'function evaluations'

        # use fake last entry in x and std for line extension-annotation
        if dat.x.shape[1] < 100:
            minxend = int(1.06*dat.x[-2, iabscissa])
            # write y-values for individual annotation into dat.x
            dat.x[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.x[-2,5:])
            idx2 = np.argsort(idx)
            if x_opt is None:
                dat.x[-1,5+idx] = np.linspace(np.min(dat.x[:,5:]),
                            np.max(dat.x[:,5:]), dat.x.shape[1]-5)
            else:
                dat.x[-1,5+idx] = np.logspace(np.log10(np.min(abs(dat.x[:,5:]))),
                            np.log10(np.max(abs(dat.x[:,5:]))), dat.x.shape[1]-5)
        else:
            minxend = 0

        if len(dat.f) == 0:
            print('nothing to plot')
            return

        # not in use anymore, see formatter above
        # xticklocs = np.arange(5) * np.round(minxend/4., -int(np.log10(minxend/4.)))

        # dfit(dfit<1e-98) = NaN;

        ioff() # turns update off

        # TODO: if abscissa==0 plot in chunks, ie loop over subsets where dat.f[:,0]==countiter is monotonous

        subplot(2,2,1)
        self.plotdivers(dat, iabscissa, foffset)

        # TODO: modularize also the remaining subplots
        subplot(2,2,2)
        hold(False)
        if x_opt is not None:  # TODO: differentate neg and pos?
            semilogy(dat.x[:, iabscissa], abs(dat.x[:,5:]) - x_opt, '-')
        else:
            plot(dat.x[:, iabscissa], dat.x[:,5:],'-')
        hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        ax[1] -= 1e-6
        if dat.x.shape[1] < 100:
            yy = np.linspace(ax[2]+1e-6, ax[3]-1e-6, dat.x.shape[1]-5)
            #yyl = np.sort(dat.x[-1,5:])
            idx = np.argsort(dat.x[-1,5:])
            idx2 = np.argsort(idx)
            if x_opt is not None:
                semilogy([dat.x[-1, iabscissa], ax[1]], [abs(dat.x[-1,5:]), yy[idx2]], 'k-') # line from last data point
                semilogy(np.dot(dat.x[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-')
            else:
                # plot([dat.x[-1, iabscissa], ax[1]], [dat.x[-1,5:], yy[idx2]], 'k-') # line from last data point
                plot(np.dot(dat.x[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-')
            # plot(array([dat.x[-1, iabscissa], ax[1]]),
            #      reshape(array([dat.x[-1,5:], yy[idx2]]).flatten(), (2,4)), '-k')
            for i in range(len(idx)):
                # TODOqqq: annotate phenotypic value!?
                # text(ax[1], yy[i], 'x(' + str(idx[i]) + ')=' + str(dat.x[-2,5+idx[i]]))
                text(dat.x[-1,iabscissa], dat.x[-1,5+i], 'x(' + str(i) + ')=' + str(dat.x[-2,5+i]))

        i = 2  # find smallest i where iteration count differs (in case the same row appears twice)
        while i < len(dat.f) and dat.f[-i][0] == dat.f[-1][0]:
            i += 1
        title('Object Variables (' + ('mean' if plot_mean else 'curr best') +
                ', ' + str(dat.x.shape[1]-5) + '-D, popsize~' +
                (str(int((dat.f[-1][1] - dat.f[-i][1]) / (dat.f[-1][0] - dat.f[-i][0])))
                    if len(dat.f.T[0]) > 1 and dat.f[-1][0] > dat.f[-i][0] else 'NA')
                + ')')
        # pylab.xticks(xticklocs)

        # Scaling
        subplot(2,2,3)
        hold(False)
        semilogy(dat.D[:, iabscissa], dat.D[:,5:], '-b')
        hold(True)
        grid(True)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        title('Scaling (All Main Axes)')
        # pylab.xticks(xticklocs)
        xlabel(xlab)

        # standard deviations
        subplot(2,2,4)
        hold(False)
        # remove sigma from stds (graphs become much better readible)
        dat.std[:,5:] = np.transpose(dat.std[:,5:].T / dat.std[:,2].T)
        # ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        # axis(ax)
        if 1 < 2 and dat.std.shape[1] < 100:
            # use fake last entry in x and std for line extension-annotation
            minxend = int(1.06*dat.x[-2, iabscissa])
            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            idx = np.argsort(dat.std[-2,5:])
            idx2 = np.argsort(idx)
            dat.std[-1,5+idx] = np.logspace(np.log10(np.min(dat.std[:,5:])),
                            np.log10(np.max(dat.std[:,5:])), dat.std.shape[1]-5)

            dat.std[-1, iabscissa] = minxend  # TODO: should be ax[1]
            yy = np.logspace(np.log10(ax[2]), np.log10(ax[3]), dat.std.shape[1]-5)
            #yyl = np.sort(dat.std[-1,5:])
            idx = np.argsort(dat.std[-1,5:])
            idx2 = np.argsort(idx)
            # plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([ax[2]+1e-6, ax[3]-1e-6]), 'k-') # vertical separator
            # vertical separator
            plot(np.dot(dat.std[-2, iabscissa],[1,1]), array([np.min(dat.std[-2,5:]), np.max(dat.std[-2,5:])]), 'k-')
            hold(True)
            # plot([dat.std[-1, iabscissa], ax[1]], [dat.std[-1,5:], yy[idx2]], 'k-') # line from last data point
            for i in xrange(len(idx)):
                # text(ax[1], yy[i], ' '+str(idx[i]))
                text(dat.std[-1, iabscissa], dat.std[-1, 5+i], ' '+str(i))
        semilogy(dat.std[:, iabscissa], dat.std[:,5:], '-')
        grid(True)
        title('Standard Deviations in All Coordinates')
        # pylab.xticks(xticklocs)
        xlabel(xlab)
        draw()  # does not suffice
        if interactive_status:
            ion()  # turns interactive mode on (again)
            draw()
        show()

        return self


    #____________________________________________________________
    #____________________________________________________________
    #
    @staticmethod
    def plotdivers(dat, iabscissa, foffset):
        """helper function for `plot()` that plots all what is
        in the upper left subplot like fitness, sigma, etc.

        Arguments
        ---------
            `iabscissa` in ``(0,1)``
                0==versus fevals, 1==versus iteration
            `foffset`
                offset to fitness for log-plot

         :See: `plot()`

        """
        from  matplotlib.pylab import semilogy, hold, grid, \
                 axis, title, text
        fontsize = pylab.rcParams['font.size']

        hold(False)

        dfit = dat.f[:,5]-min(dat.f[:,5])
        dfit[dfit<1e-98] = np.NaN

        if dat.f.shape[1] > 7:
            # semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7, 10, 12]])+foffset,'-k')
            semilogy(dat.f[:, iabscissa], abs(dat.f[:,[6, 7]])+foffset,'-k')
            hold(True)

        # (larger indices): additional fitness data, for example constraints values
        if dat.f.shape[1] > 8:
            # dd = abs(dat.f[:,7:]) + 10*foffset
            # dd = np.where(dat.f[:,7:]==0, np.NaN, dd) # cannot be
            semilogy(dat.f[:, iabscissa], np.abs(dat.f[:,8:]) + 10*foffset, 'm')
            hold(True)

        idx = np.where(dat.f[:,5]>1e-98)[0]  # positive values
        semilogy(dat.f[idx, iabscissa], dat.f[idx,5]+foffset, '.b')
        hold(True)
        grid(True)

        idx = np.where(dat.f[:,5] < -1e-98)  # negative values
        semilogy(dat.f[idx, iabscissa], abs(dat.f[idx,5])+foffset,'.r')

        semilogy(dat.f[:, iabscissa],abs(dat.f[:,5])+foffset,'-b')
        semilogy(dat.f[:, iabscissa], dfit, '-c')

        if 11 < 3:  # delta-fitness as points
            dfit = dat.f[1:, 5] - dat.f[:-1,5]  # should be negative usually
            semilogy(dat.f[1:,iabscissa],  # abs(fit(g) - fit(g-1))
                np.abs(dfit)+foffset, '.c')
            i = dfit > 0
            # print(np.sum(i) / float(len(dat.f[1:,iabscissa])))
            semilogy(dat.f[1:,iabscissa][i],  # abs(fit(g) - fit(g-1))
                np.abs(dfit[i])+foffset, '.r')

        # overall minimum
        i = np.argmin(dat.f[:,5])
        semilogy(dat.f[i, iabscissa]*np.ones(2), dat.f[i,5]*np.ones(2), 'rd')
        # semilogy(dat.f[-1, iabscissa]*np.ones(2), dat.f[-1,4]*np.ones(2), 'rd')

        # AR and sigma
        semilogy(dat.f[:, iabscissa], dat.f[:,3], '-r') # AR
        semilogy(dat.f[:, iabscissa], dat.f[:,2],'-g') # sigma
        semilogy(dat.std[:-1, iabscissa], np.vstack([list(map(max, dat.std[:-1,5:])), list(map(min, dat.std[:-1,5:]))]).T,
                     '-m', linewidth=2)
        text(dat.std[-2, iabscissa], max(dat.std[-2, 5:]), 'max std', fontsize=fontsize)
        text(dat.std[-2, iabscissa], min(dat.std[-2, 5:]), 'min std', fontsize=fontsize)
        ax = array(axis())
        # ax[1] = max(minxend, ax[1])
        axis(ax)
        text(ax[0]+0.01, ax[2], # 10**(log10(ax[2])+0.05*(log10(ax[3])-log10(ax[2]))),
             '.f_recent=' + repr(dat.f[-1,5]) )

        # title('abs(f) (blue), f-min(f) (cyan), Sigma (green), Axis Ratio (red)')
        title('blue:abs(f), cyan:f-min(f), green:sigma, red:axis ratio', fontsize=fontsize-1)
        # pylab.xticks(xticklocs)


    def downsampling(self, factor=10, first=3, switch=True):
        """
        rude downsampling of a `CMADataLogger` data file by `factor`, keeping
        also the first `first` entries. This function is a stump and subject
        to future changes.

        Arguments
        ---------
           - `factor` -- downsampling factor
           - `first` -- keep first `first` entries
           - `switch` -- switch the new logger name to oldname+'down'

        Details
        -------
        ``self.name_prefix+'down'`` files are written

        Example
        -------
        ::

            import cma
            cma.downsampling()  # takes outcmaes* files
            cma.plot('outcmaesdown')

        """
        newprefix = self.name_prefix + 'down'
        for name in CMADataLogger.names:
            f = open(newprefix+name+'.dat','w')
            iline = 0
            cwritten = 0
            for line in open(self.name_prefix+name+'.dat'):
                if iline < first or iline % factor == 0:
                    f.write(line)
                    cwritten += 1
                iline += 1
            f.close()
            print('%d' % (cwritten) + ' lines written in ' + newprefix+name+'.dat')
        if switch:
            self.name_prefix += 'down'
        return self

    #____________________________________________________________
    #____________________________________________________________
    #
    def disp_header(self):
        heading = 'Iterat Nfevals  function value    axis ratio maxstd   minstd'
        print(heading)

    def disp(self, idx=100):  # r_[0:5,1e2:1e9:1e2,-10:0]):
        """displays selected data from (files written by) the class `CMADataLogger`.

        Arguments
        ---------
           `idx`
               indices corresponding to rows in the data file;
               if idx is a scalar (int), the first two, then every idx-th,
               and the last three rows are displayed. Too large index values are removed.
               If ``len(idx) == 1``, only a single row is displayed, e.g. the last
               entry when ``idx == [-1]``.

        Example
        -------
        >>> import cma, numpy as np
        >>> res = cma.fmin(cma.fcts.elli, 7 * [0.1], 1, verb_disp=1e9)  # generate data
        >>> assert res[1] < 1e-9
        >>> assert res[2] < 4400
        >>> l = cma.CMADataLogger()  # == res[-1], logger with default name, "points to" above data
        >>> l.disp([0,-1])  # first and last
        >>> l.disp(20)  # some first/last and every 20-th line
        >>> l.disp(np.r_[0:999999:100, -1]) # every 100-th and last
        >>> l.disp(np.r_[0, -10:0]) # first and ten last
        >>> cma.disp(l.name_prefix, np.r_[0::100, -10:])  # the same as l.disp(...)

        Details
        -------
        The data line with the best f-value is displayed as last line.

        :See: `disp()`

        """

        filenameprefix=self.name_prefix

        def printdatarow(dat, iteration):
            """print data of iteration i"""
            i = np.where(dat.f[:, 0] == iteration)[0][0]
            j = np.where(dat.std[:, 0] == iteration)[0][0]
            print('%5d' % (int(dat.f[i,0])) + ' %6d' % (int(dat.f[i,1])) + ' %.14e' % (dat.f[i,5]) +
                  ' %5.1e' % (dat.f[i,3]) +
                  ' %6.2e' % (max(dat.std[j,5:])) + ' %6.2e' % min(dat.std[j,5:]))

        dat = CMADataLogger(filenameprefix).load()
        ndata = dat.f.shape[0]

        # map index to iteration number, is difficult if not all iteration numbers exist
        # idx = idx[np.where(map(lambda x: x in dat.f[:,0], idx))[0]] # TODO: takes pretty long
        # otherwise:
        if idx is None:
            idx = 100
        if np.isscalar(idx):
            # idx = np.arange(0, ndata, idx)
            if idx:
                idx = np.r_[0, 1, idx:ndata-3:idx, -3:0]
            else:
                idx = np.r_[0, 1, -3:0]

        idx = array(idx)
        idx = idx[idx<=ndata]  # TODO: shouldn't this be "<"?
        idx = idx[-idx<=ndata]
        iters = dat.f[idx, 0]
        idxbest = np.argmin(dat.f[:,5])
        iterbest = dat.f[idxbest, 0]
        if len(iters) == 1:
            printdatarow(dat, iters[0])
        else:
            self.disp_header()
            for i in iters:
                printdatarow(dat, i)
            self.disp_header()
            printdatarow(dat, iterbest)
        sys.stdout.flush()

def irg(ar):
    return xrange(len(ar))
class AII(object):
    """unstable experimental code, updates ps, sigma, sigmai, pr, r, sigma_r, mean,
    all from self.

    Depends on that the ordering of solutions has not change upon calling update

    should become a OOOptimizer in far future?

    """
    # Try: ps**2 - 1 instead of (ps**2)**0.5 / chi1 - 1: compare learning rate etc
    # and dito for psr

    def __init__(self, x0, sigma0, randn=np.random.randn):
        """TODO: check scaling of r-learing: seems worse than linear: 9e3 25e3 65e3 (10,20,40-D)"""
        self.N = len(x0)
        N = self.N
        # parameters to play with:
        # PROBLEM: smaller eta_r even fails on *axparallel* cigar!! Also dampi needs to be smaller then!
        self.dampi = 4 * N  # two times smaller is
        self.eta_r = 0 / N / 3   # c_r learning rate for direction, cigar: 4/N/3 is optimal in 10-D, 10/N/3 still works (15 in 20-D) but not on the axparallel cigar with recombination
        self.mu = 1
        self.use_abs_sigma = 1    # without it is a problem on 20=D axpar-cigar!!, but why?? Because dampi is just boarderline
        self.use_abs_sigma_r = 1  #

        self.randn = randn
        self.x0 = array(x0, copy=True)
        self.sigma0 = sigma0

        self.cs = 1 / N**0.5  # evolution path for step-size(s)
        self.damps = 1
        self.use_sign = 0
        self.use_scalar_product = 0  # sometimes makes it somewhat worse on Rosenbrock, don't know why
        self.csr = 1 / N**0.5  # cumulation for sigma_r
        self.dampsr = (4 * N)**0.5
        self.chi1 = (2/np.pi)**0.5
        self.chiN = N**0.5*(1-1./(4.*N)+1./(21.*N**2)) # expectation of norm(randn(N,1))
        self.initialize()
    def initialize(self):
        """alias ``reset``, set all state variables to initial values"""
        N = self.N
        self.mean = array(self.x0, copy=True)
        self.sigma = self.sigma0
        self.sigmai = np.ones(N)
        self.ps = np.zeros(N)  # path for individual and globalstep-size(s)
        self.r = np.zeros(N)
        self.pr = 0         # cumulation for zr = N(0,1)
        self.sigma_r = 0
    def ask(self, popsize):
        if popsize == 1:
            raise NotImplementedError()
        self.Z = [self.randn(self.N) for _i in xrange(popsize)]
        self.zr = list(self.randn(popsize))
        pop = [self.mean + self.sigma * (self.sigmai * self.Z[k])
                + self.zr[k] * self.sigma_r * self.r
                for k in xrange(popsize)]
        if not np.isfinite(pop[0][0]):
            raise ValueError()
        return pop
    def tell(self, X, f):
        """update """
        mu = 1 if self.mu else int(len(f) / 4)
        idx = np.argsort(f)[:mu]
        zr = [self.zr[i] for i in idx]
        Z = [self.Z[i] for i in idx]
        X = [X[i] for i in idx]
        xmean = np.mean(X, axis=0)

        self.ps *= 1 - self.cs
        self.ps += (self.cs*(2-self.cs))**0.5 * mu**0.5 * np.mean(Z, axis=0)
        self.sigma *= np.exp((self.cs/self.damps) * (sum(self.ps**2)**0.5 / self.chiN - 1))
        if self.use_abs_sigma:
            self.sigmai *= np.exp((1/self.dampi) * (np.abs(self.ps) / self.chi1 - 1))
        else:
            self.sigmai *= np.exp((1.3/self.dampi/2) * (self.ps**2 - 1))

        self.pr *= 1 - self.csr
        self.pr += (self.csr*(2-self.csr))**0.5 * mu**0.5 * np.mean(zr)
        fac = 1
        if self.use_sign:
            fac = np.sign(self.pr)  # produces readaptations on the cigar
        else:
            self.pr = max([0, self.pr])
        if self.use_scalar_product:
            if np.sign(sum(self.r * (xmean - self.mean))) < 0: # and self.pr > 1:
            # if np.sign(sum(self.r * self.ps)) < 0:
                self.r *= -1
        if self.eta_r:
            self.r *= (1 - self.eta_r) * self.sigma_r
            self.r += fac * self.eta_r * mu**0.5 * (xmean - self.mean)
            self.r /= sum(self.r**2)**0.5
        if self.use_abs_sigma_r:
            self.sigma_r *= np.exp((1/self.dampsr) * ((self.pr**2)**0.5 / self.chi1 - 1))
        else:
            # this is worse on the cigar, where the direction vector(!) behaves strangely
            self.sigma_r *= np.exp((1/self.dampsr) * (self.pr**2 - 1) / 2)
        self.sigma_r = max([self.sigma * sum(self.sigmai**2)**0.5 / 3, self.sigma_r])
        # self.sigma_r = 0
        self.mean = xmean
def fmin(func, x0, sigma0=None, args=()
    # the follow string arguments are evaluated, besides the verb_filenameprefix
    , CMA_active='False  # exponential negative update, conducted after the original update'
    , CMA_activefac='1  # learning rate multiplier for active update'
    , CMA_cmean='1  # learning rate for the mean value'
    , CMA_const_trace='False  # normalize trace, value CMA_const_trace=2 normalizes sum log eigenvalues to zero'
    , CMA_diagonal='0*100*N/sqrt(popsize)  # nb of iterations with diagonal covariance matrix, True for always' # TODO 4/ccov_separable?
    , CMA_eigenmethod='np.linalg.eigh  # 0=numpy-s eigh, -1=pygsl, otherwise cma.Misc.eig (slower)'
    , CMA_elitist='False # elitism likely impairs global search performance'
    , CMA_mirrors='popsize < 6  # values <0.5 are interpreted as fraction, values >1 as numbers (rounded), otherwise about 0.16 is used'
    , CMA_mu='None  # parents selection parameter, default is popsize // 2'
    , CMA_on='True  # False or 0 for no adaptation of the covariance matrix'
    , CMA_rankmu='True  # False or 0 for omitting rank-mu update of covariance matrix'
    , CMA_rankmualpha='0.3  # factor of rank-mu update if mu=1, subject to removal, default might change to 0.0'
    , CMA_dampfac='1  #v positive multiplier for step-size damping, 0.3 is close to optimal on the sphere'
    , CMA_dampsvec_fac='np.Inf  # tentative and subject to changes, 0.5 would be a "default" damping for sigma vector update'
    , CMA_dampsvec_fade='0.1  # tentative fading out parameter for sigma vector update'
    , CMA_teststds='None  # factors for non-isotropic initial distr. mainly for test purpose, see scaling_...'
    , CMA_AII='False  # not yet tested'
    , bounds='[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a scalar or a list/vector'
    , eval_parallel='False  # when True, func might be called with more than one solution as first argument'
    , eval_initial_x='False  # '
    , fixed_variables='None  # dictionary with index-value pairs like {0:1.1, 2:0.1} that are not optimized'
    , ftarget='-inf  #v target function value, minimization'
    , incpopsize='2  # in fmin(): multiplier for increasing popsize before each restart'
    , maxfevals='inf  #v maximum number of function evaluations'
    , maxiter='100 + 50 * (N+3)**2 // popsize**0.5  #v maximum number of iterations'
    , mindx='0  #v minimal std in any direction, cave interference with tol*'
    , minstd='0  #v minimal std in any coordinate direction, cave interference with tol*'
    , noise_handling='False  # maximal number of evaluations for noise treatment, only fmin'
    , noise_reevals=' 1.5 + popsize/20  # number of solution to be reevaluated for noise measurement, only fmin'
    , noise_eps='1e-7  # perturbation factor for noise handling reevaluations, only fmin'
    , noise_change_sigma='True  # exponent to default sigma increment'
    , popsize='4+int(3*log(N))  # population size, AKA lambda, number of new solution per iteration'
    , randn='np.random.standard_normal  #v randn((lam, N)) must return an np.array of shape (lam, N)'
    , restarts='0  # in fmin(): number of restarts'
    , restart_from_best='False'
    , scaling_of_variables='None  # scale for each variable, sigma0 is interpreted w.r.t. this scale, in that effective_sigma0 = sigma0*scaling. Internally the variables are divided by scaling_of_variables and sigma is unchanged, default is ones(N)'
    , seed='None  # random number seed'
    , termination_callback='None  #v a function returning True for termination, called after each iteration step and could be abused for side effects'
    , tolfacupx='1e3  #v termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen far too small and better solutions were found far away from the initial solution x0'
    , tolupsigma='1e20  #v sigma/sigma0 > tolupsigma * max(sqrt(eivenvals(C))) indicates "creeping behavior" with usually minor improvements'
    , tolfun='1e-11  #v termination criterion: tolerance in function value, quite useful'
    , tolfunhist='1e-12  #v termination criterion: tolerance in function value history'
    , tolstagnation='int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations'
    , tolx='1e-11  #v termination criterion: tolerance in x-changes'
    , transformation='None  # [t0, t1] are two mappings, t0 transforms solutions from CMA-representation to f-representation (tf_pheno), t1 is the (optional) back transformation, see class GenoPheno'
    , typical_x='None  # used with scaling_of_variables'
    , updatecovwait='None  #v number of iterations without distribution update, name is subject to future changes' # TODO: rename: iterwaitupdatedistribution?
    , verb_append='0  # initial evaluation counter, if append, do not overwrite output files'
    , verb_disp='100  #v verbosity: display console output every verb_disp iteration'
    , verb_filenameprefix='outcmaes  # output filenames prefix'
    , verb_log='1  #v verbosity: write data to files every verb_log iteration, writing can be time critical on fast to evaluate functions'
    , verb_plot='0  #v in fmin(): plot() is called every verb_plot iteration'
    , verb_time='True  #v output timings on console'
    , vv='0  #? versatile variable for hacking purposes, value found in self.opts[\'vv\']'
     ):
    """functional interface to the stochastic optimizer CMA-ES
    for non-convex function minimization.

    Calling Sequences
    =================
        ``fmin([],[])``
            returns all optional arguments, that is,
            all keyword arguments to fmin with their default values
            in a dictionary.
        ``fmin(func, x0, sigma0)``
            minimizes `func` starting at `x0` and with standard deviation
            `sigma0` (step-size)
        ``fmin(func, x0, sigma0, ftarget=1e-5)``
            minimizes `func` up to target function value 1e-5
        ``fmin(func, x0, sigma0, args=('f',), **options)``
            minimizes `func` called with an additional argument ``'f'``.
            `options` is a dictionary with additional keyword arguments, e.g.
            delivered by `Options()`.
        ``fmin(func, x0, sigma0, **{'ftarget':1e-5, 'popsize':40})``
            the same as ``fmin(func, x0, sigma0, ftarget=1e-5, popsize=40)``
        ``fmin(func, esobj, **{'maxfevals': 1e5})``
            uses the `CMAEvolutionStrategy` object instance `esobj` to optimize
            `func`, similar to `CMAEvolutionStrategy.optimize()`.

    Arguments
    =========
        `func`
            function to be minimized. Called as
            ``func(x,*args)``. `x` is a one-dimensional `numpy.ndarray`. `func`
            can return `numpy.NaN`,
            which is interpreted as outright rejection of solution `x`
            and invokes an immediate resampling and (re-)evaluation
            of a new solution not counting as function evaluation.
        `x0`
            list or `numpy.ndarray`, initial guess of minimum solution
            or `cma.CMAEvolutionStrategy` object instance. In this case
            `sigma0` can be omitted.
        `sigma0`
            scalar, initial standard deviation in each coordinate.
            `sigma0` should be about 1/4 of the search domain width where the
            optimum is to be expected. The variables in `func` should be
            scaled such that they presumably have similar sensitivity.
            See also option `scaling_of_variables`.

    Keyword Arguments
    =================
    All arguments besides `args` and `verb_filenameprefix` are evaluated
    if they are of type `str`, see class `Options` for details. The following
    list might not be fully up-to-date, use ``cma.Options()`` or
    ``cma.fmin([],[])`` to get the actual list.
    ::

        args=() -- additional arguments for func, not in `cma.Options()`
        CMA_active='False  # exponential negative update, conducted after the original
                update'
        CMA_activefac='1  # learning rate multiplier for active update'
        CMA_cmean='1  # learning rate for the mean value'
        CMA_dampfac='1  #v positive multiplier for step-size damping, 0.3 is close to
                optimal on the sphere'
        CMA_diagonal='0*100*N/sqrt(popsize)  # nb of iterations with diagonal
                covariance matrix, True for always'
        CMA_eigenmethod='np.linalg.eigh  # 0=numpy-s eigh, -1=pygsl, alternative: Misc.eig (slower)'
        CMA_elitist='False # elitism likely impairs global search performance'
        CMA_mirrors='0  # values <0.5 are interpreted as fraction, values >1 as numbers
                (rounded), otherwise about 0.16 is used'
        CMA_mu='None  # parents selection parameter, default is popsize // 2'
        CMA_on='True  # False or 0 for no adaptation of the covariance matrix'
        CMA_rankmu='True  # False or 0 for omitting rank-mu update of covariance
                matrix'
        CMA_rankmualpha='0.3  # factor of rank-mu update if mu=1, subject to removal,
                default might change to 0.0'
        CMA_teststds='None  # factors for non-isotropic initial distr. mainly for test
                purpose, see scaling_...'
        bounds='[None, None]  # lower (=bounds[0]) and upper domain boundaries, each a
                scalar or a list/vector'
        eval_initial_x='False  # '
        fixed_variables='None  # dictionary with index-value pairs like {0:1.1, 2:0.1}
                that are not optimized'
        ftarget='-inf  #v target function value, minimization'
        incpopsize='2  # in fmin(): multiplier for increasing popsize before each
                restart'
        maxfevals='inf  #v maximum number of function evaluations'
        maxiter='long(1e3*N**2/sqrt(popsize))  #v maximum number of iterations'
        mindx='0  #v minimal std in any direction, cave interference with tol*'
        minstd='0  #v minimal std in any coordinate direction, cave interference with
                tol*'
        noise_eps='1e-7  # perturbation factor for noise handling reevaluations, only
                fmin'
        noise_handling='False  # maximal number of evaluations for noise treatment,
                only fmin'
        noise_reevals=' 1.5 + popsize/20  # number of solution to be reevaluated for
                noise measurement, only fmin'
        popsize='4+int(3*log(N))  # population size, AKA lambda, number of new solution
                per iteration'
        randn='np.random.standard_normal  #v randn((lam, N)) must return an np.array of
                shape (lam, N)'
        restarts='0  # in fmin(): number of restarts'
        scaling_of_variables='None  # scale for each variable, sigma0 is interpreted
                w.r.t. this scale, in that effective_sigma0 = sigma0*scaling.
                Internally the variables are divided by scaling_of_variables and sigma
                is unchanged, default is ones(N)'
        seed='None  # random number seed'
        termination_callback='None  #v in fmin(): a function returning True for
                termination, called after each iteration step and could be abused for
                side effects'
        tolfacupx='1e3  #v termination when step-size increases by tolfacupx
                (diverges). That is, the initial step-size was chosen far too small and
                better solutions were found far away from the initial solution x0'
        tolupsigma='1e20  #v sigma/sigma0 > tolupsigma * max(sqrt(eivenvals(C)))
                indicates "creeping behavior" with usually minor improvements'
        tolfun='1e-11  #v termination criterion: tolerance in function value, quite
                useful'
        tolfunhist='1e-12  #v termination criterion: tolerance in function value
                history'
        tolstagnation='int(100 * N**1.5 / popsize)  #v termination if no improvement
                over tolstagnation iterations'
        tolx='1e-11  #v termination criterion: tolerance in x-changes'
        transformation='None  # [t0, t1] are two mappings, t0 transforms solutions from
                CMA-representation to f-representation, t1 is the back transformation,
                see class GenoPheno'
        typical_x='None  # used with scaling_of_variables'
        updatecovwait='None  #v number of iterations without distribution update, name
                is subject to future changes'
        verb_append='0  # initial evaluation counter, if append, do not overwrite
                output files'
        verb_disp='100  #v verbosity: display console output every verb_disp iteration'
        verb_filenameprefix='outcmaes  # output filenames prefix'
        verb_log='1  #v verbosity: write data to files every verb_log iteration,
                writing can be time critical on fast to evaluate functions'
        verb_plot='0  #v in fmin(): plot() is called every verb_plot iteration'
        verb_time='True  #v output timings on console'
        vv='0  #? versatile variable for hacking purposes, value found in
                self.opts['vv']'

    Subsets of options can be displayed, for example like ``cma.Options('tol')``,
    see also class `Options`.

    Return
    ======
    Similar to `OOOptimizer.optimize()` and/or `CMAEvolutionStrategy.optimize()`, return the
    list provided by `CMAEvolutionStrategy.result()` appended with an `OOOptimizer` and an
    `BaseDataLogger`::

        res = optim.result() + (optim.stop(), optim, logger)

    where
        - ``res[0]`` (``xopt``) -- best evaluated solution
        - ``res[1]`` (``fopt``) -- respective function value
        - ``res[2]`` (``evalsopt``) -- respective number of function evaluations
        - ``res[3]`` (``evals``) -- number of overall conducted objective function evaluations
        - ``res[4]`` (``iterations``) -- number of overall conducted iterations
        - ``res[5]`` (``xmean``) -- mean of the final sample distribution
        - ``res[6]`` (``stds``) -- effective stds of the final sample distribution
        - ``res[-3]`` (``stop``) -- termination condition(s) in a dictionary
        - ``res[-2]`` (``cmaes``) -- class `CMAEvolutionStrategy` instance
        - ``res[-1]`` (``logger``) -- class `CMADataLogger` instance

    Details
    =======
    This function is an interface to the class `CMAEvolutionStrategy`. The
    class can be used when full control over the iteration loop of the
    optimizer is desired.

    The noise handling follows closely [Hansen et al 2009, A Method for Handling
    Uncertainty in Evolutionary Optimization...] in the measurement part, but the
    implemented treatment is slightly different: for ``noiseS > 0``, ``evaluations``
    (time) and sigma are increased by ``alpha``. For ``noiseS < 0``, ``evaluations``
    (time) is decreased by ``alpha**(1/4)``. The option ``noise_handling`` switches
    the uncertainty handling on/off, the given value defines the maximal number
    of evaluations for a single fitness computation. If ``noise_handling`` is a list,
    the smallest element defines the minimal number and if the list has three elements,
    the median value is the start value for ``evaluations``. See also class
    `NoiseHandler`.

    Examples
    ========
    The following example calls `fmin` optimizing the Rosenbrock function
    in 10-D with initial solution 0.1 and initial step-size 0.5. The
    options are specified for the usage with the `doctest` module.

    >>> import cma
    >>> # cma.Options()  # returns all possible options
    >>> options = {'CMA_diagonal':10, 'seed':1234, 'verb_time':0}
    >>>
    >>> res = cma.fmin(cma.fcts.rosen, [0.1] * 10, 0.5, **options)
    (5_w,10)-CMA-ES (mu_w=3.2,w_1=45%) in dimension 10 (seed=1234)
       Covariance matrix is diagonal for 10 iterations (1/ccov=29.0)
    Iterat #Fevals   function value     axis ratio  sigma   minstd maxstd min:sec
        1      10 1.264232686260072e+02 1.1e+00 4.40e-01  4e-01  4e-01
        2      20 1.023929748193649e+02 1.1e+00 4.00e-01  4e-01  4e-01
        3      30 1.214724267489674e+02 1.2e+00 3.70e-01  3e-01  4e-01
      100    1000 6.366683525319511e+00 6.2e+00 2.49e-02  9e-03  3e-02
      200    2000 3.347312410388666e+00 1.2e+01 4.52e-02  8e-03  4e-02
      300    3000 1.027509686232270e+00 1.3e+01 2.85e-02  5e-03  2e-02
      400    4000 1.279649321170636e-01 2.3e+01 3.53e-02  3e-03  3e-02
      500    5000 4.302636076186532e-04 4.6e+01 4.78e-03  3e-04  5e-03
      600    6000 6.943669235595049e-11 5.1e+01 5.41e-06  1e-07  4e-06
      650    6500 5.557961334063003e-14 5.4e+01 1.88e-07  4e-09  1e-07
    termination on tolfun : 1e-11
    final/bestever f-value = 5.55796133406e-14 2.62435631419e-14
    mean solution:  [ 1.          1.00000001  1.          1.
        1.          1.00000001  1.00000002  1.00000003 ...]
    std deviation: [ 3.9193387e-09  3.7792732e-09  4.0062285e-09  4.6605925e-09
        5.4966188e-09   7.4377745e-09   1.3797207e-08   2.6020765e-08 ...]
    >>>
    >>> print('best solutions fitness = %f' % (res[1]))
    best solutions fitness = 2.62435631419e-14
    >>> assert res[1] < 1e-12

    The method ::

        cma.plot();

    (based on `matplotlib.pylab`) produces a plot of the run and, if necessary::

        cma.show()

    shows the plot in a window. To continue you might need to
    close the pop-up window. This behavior seems to disappear in
    subsequent calls of `cma.plot()` and is avoided by using
    `ipython` with `-pylab` option. Finally ::

        cma.savefig('myfirstrun')  # savefig from matplotlib.pylab

    will save the figure in a png.

    :See: `CMAEvolutionStrategy`, `OOOptimizer.optimize(), `plot()`, `Options`, `scipy.optimize.fmin()`

    """ # style guides say there should be the above empty line
    try: # pass on KeyboardInterrupt
        opts = locals()  # collect all local variables (i.e. arguments) in a dictionary
        del opts['func'] # remove those without a default value
        del opts['args']
        del opts['x0']      # is not optional, no default available
        del opts['sigma0']  # is not optional for the constructor CMAEvolutionStrategy
        if not func:  # return available options in a dictionary
            return Options(opts, True)  # these opts are by definition valid

        # TODO: this is very ugly:
        incpopsize = Options({'incpopsize':incpopsize}).eval('incpopsize')
        restarts = Options({'restarts':restarts}).eval('restarts')
        del opts['restarts']
        noise_handling = Options({'noise_handling': noise_handling}).eval('noise_handling')
        del opts['noise_handling']# otherwise CMA throws an error

        irun = 0
        best = BestSolution()
        while 1:
            # recover from a CMA object
            if irun == 0 and isinstance(x0, CMAEvolutionStrategy):
                es = x0
                x0 = es.inputargs['x0']  # for the next restarts
                if sigma0 is None or not np.isscalar(array(sigma0)):
                    sigma0 = es.inputargs['sigma0']  # for the next restarts
                # ignore further input args and keep original options
            else:  # default case
                if irun and opts['restart_from_best']:
                    print('CAVE: restart_from_best is typically not useful')
                    es = CMAEvolutionStrategy(best.x, sigma0, opts)
                else:
                    es = CMAEvolutionStrategy(x0, sigma0, opts)
                if opts['eval_initial_x']:
                    x = es.gp.pheno(es.mean, bounds=es.gp.bounds)
                    es.best.update([x], None, [func(x, *args)], 1)
                    es.countevals += 1

            opts = es.opts  # processed options, unambiguous

            append = opts['verb_append'] or es.countiter > 0 or irun > 0
            logger = CMADataLogger(opts['verb_filenameprefix'], opts['verb_log'])
            logger.register(es, append).add()  # initial values, not fitness values

            # if es.countiter == 0 and es.opts['verb_log'] > 0 and not es.opts['verb_append']:
            #    logger = CMADataLogger(es.opts['verb_filenameprefix']).register(es)
            #    logger.add()
            # es.writeOutput()  # initial values for sigma etc

            noisehandler = NoiseHandler(es.N, noise_handling, np.median, opts['noise_reevals'], opts['noise_eps'], opts['eval_parallel'])
            while not es.stop():
                X, fit = es.ask_and_eval(func, args, evaluations=noisehandler.evaluations,
                                         aggregation=np.median) # treats NaN with resampling
                # TODO: check args and in case use args=(noisehandler.evaluations, )

                if 11 < 3 and opts['vv']:  # inject a solution
                    # use option check_point = [0]
                    if 0 * np.random.randn() >= 0:
                        X[0] = 0 + opts['vv'] * es.sigma**0 * np.random.randn(es.N)
                        fit[0] = func(X[0], *args)
                        # print fit[0]
                es.tell(X, fit)  # prepare for next iteration
                if noise_handling:
                    es.sigma *= noisehandler(X, fit, func, es.ask, args)**opts['noise_change_sigma']
                    es.countevals += noisehandler.evaluations_just_done  # TODO: this is a hack, not important though

                es.disp()
                logger.add(more_data=[noisehandler.evaluations, 10**noisehandler.noiseS] if noise_handling else [],
                           modulo=1 if es.stop() and logger.modulo else None)
                if opts['verb_log'] and opts['verb_plot'] and \
                    (es.countiter % max(opts['verb_plot'], opts['verb_log']) == 0 or es.stop()):
                    logger.plot(324, fontsize=10)

            # end while not es.stop
            mean_pheno = es.gp.pheno(es.mean, bounds=es.gp.bounds)
            fmean = func(mean_pheno, *args)
            es.countevals += 1

            es.best.update([mean_pheno], None, [fmean], es.countevals)
            best.update(es.best)  # in restarted case

            # final message
            if opts['verb_disp']:
                srestarts = (' after %i restart' + ('s' if irun > 1 else '')) % irun if irun else ''
                for k, v in list(es.stop().items()):
                    print('termination on %s=%s%s (%s)' % (k, str(v), srestarts, time.asctime()))

                print('final/bestever f-value = %e %e' % (es.best.last.f, best.f))
                if es.N < 9:
                    print('mean solution: ' + str(es.gp.pheno(es.mean)))
                    print('std deviation: ' + str(es.sigma * sqrt(es.dC) * es.gp.scales))
                else:
                    print('mean solution: %s ...]' % (str(es.gp.pheno(es.mean)[:8])[:-1]))
                    print('std deviations: %s ...]' % (str((es.sigma * sqrt(es.dC) * es.gp.scales)[:8])[:-1]))

            irun += 1
            if irun > restarts or 'ftarget' in es.stopdict or 'maxfunevals' in es.stopdict:
                break
            opts['verb_append'] = es.countevals
            opts['popsize'] = incpopsize * es.sp.popsize # TODO: use rather options?
            opts['seed'] += 1

        # while irun

        es.out['best'] = best  # TODO: this is a rather suboptimal type for inspection in the shell
        if 1 < 3:
            return es.result() + (es.stop(), es, logger)

        else: # previously: to be removed
            return (best.x.copy(), best.f, es.countevals,
                    dict((('stopdict', CMAStopDict(es.stopdict))
                          ,('mean', es.gp.pheno(es.mean))
                          ,('std', es.sigma * sqrt(es.dC) * es.gp.scales)
                          ,('out', es.out)
                          ,('opts', es.opts)  # last state of options
                          ,('cma', es)
                          ,('inputargs', es.inputargs)
                          ))
                   )
        # TODO refine output, can #args be flexible?
        # is this well usable as it is now?
    except KeyboardInterrupt:  # Exception, e:
        if opts['verb_disp'] > 0:
            print(' in/outcomment ``raise`` in last line of cma.fmin to prevent/restore KeyboardInterrupt exception')
        raise  # cave: swallowing this exception can silently mess up experiments, if ctrl-C is hit
def plot(name=None, fig=None, abscissa=1, iteridx=None, plot_mean=True,  # TODO: plot_mean default should be False
    foffset=1e-19, x_opt=None, fontsize=10):
    """
    plot data from files written by a `CMADataLogger`,
    the call ``cma.plot(name, **argsdict)`` is a shortcut for
    ``cma.CMADataLogger(name).plot(**argsdict)``

    Arguments
    ---------
        `name`
            name of the logger, filename prefix, None evaluates to
            the default 'outcmaes'
        `fig`
            filename or figure number, or both as a tuple (any order)
        `abscissa`
            0==plot versus iteration count,
            1==plot versus function evaluation number
        `iteridx`
            iteration indices to plot

    Return `None`

    Examples
    --------
    ::

       cma.plot();  # the optimization might be still
                    # running in a different shell
       cma.show()  # to continue you might need to close the pop-up window
                   # once and call cma.plot() again.
                   # This behavior seems to disappear in subsequent
                   # calls of cma.plot(). Also using ipython with -pylab
                   # option might help.
       cma.savefig('fig325.png')
       cma.close()

       cdl = cma.CMADataLogger().downsampling().plot()

    Details
    -------
    Data from codes in other languages (C, Java, Matlab, Scilab) have the same
    format and can be plotted just the same.

    :See: `CMADataLogger`, `CMADataLogger.plot()`

    """
    CMADataLogger(name).plot(fig, abscissa, iteridx, plot_mean, foffset, x_opt, fontsize)
def disp(name=None, idx=None):
    """displays selected data from (files written by) the class `CMADataLogger`.

    The call ``cma.disp(name, idx)`` is a shortcut for ``cma.CMADataLogger(name).disp(idx)``.

    Arguments
    ---------
        `name`
            name of the logger, filename prefix, `None` evaluates to
            the default ``'outcmaes'``
        `idx`
            indices corresponding to rows in the data file; by
            default the first five, then every 100-th, and the last
            10 rows. Too large index values are removed.

    Examples
    --------
    ::

       import cma, numpy
       # assume some data are available from previous runs
       cma.disp(None,numpy.r_[0,-1])  # first and last
       cma.disp(None,numpy.r_[0:1e9:100,-1]) # every 100-th and last
       cma.disp(idx=numpy.r_[0,-10:0]) # first and ten last
       cma.disp(idx=numpy.r_[0:1e9:1e3,-10:0])

    :See: `CMADataLogger.disp()`

    """
    return CMADataLogger(name if name else 'outcmaes'
                         ).disp(idx)

#____________________________________________________________
def _fileToMatrix(file_name):
    """rudimentary method to read in data from a file"""
    # TODO: np.loadtxt() might be an alternative
    #     try:
    if 1 < 3:
        lres = []
        for line in open(file_name, 'r').readlines():
            if len(line) > 0 and line[0] not in ('%', '#'):
                lres.append(list(map(float, line.split())))
        res = lres
    else:
        fil = open(file_name, 'r')
        fil.readline() # rudimentary, assume one comment line
        lineToRow = lambda line: list(map(float, line.split()))
        res = list(map(lineToRow, fil.readlines()))
        fil.close()  # close file could be omitted, reference counting should do during garbage collection, but...

    while res != [] and res[0] == []:  # remove further leading empty lines
        del res[0]
    return res
    #     except:
    print('could not read file ' + file_name)

#____________________________________________________________
#____________________________________________________________
class NoiseHandler(object):
    """Noise handling according to [Hansen et al 2009, A Method for Handling
    Uncertainty in Evolutionary Optimization...]

    The interface of this class is yet versatile and subject to changes.

    The attribute ``evaluations`` serves to control the noise via number of
    evaluations, for example with `ask_and_eval()`, see also parameter
    ``maxevals`` and compare the example.

    Example
    -------
    >>> import cma, numpy as np
    >>> func = cma.Fcts.noisysphere
    >>> es = cma.CMAEvolutionStrategy(np.ones(10), 1)
    >>> logger = cma.CMADataLogger().register(es)
    >>> nh = cma.NoiseHandler(es.N, maxevals=[1, 30])
    >>> while not es.stop():
    ...     X, fit = es.ask_and_eval(func, evaluations=nh.evaluations)
    ...     es.tell(X, fit)  # prepare for next iteration
    ...     es.sigma *= nh(X, fit, func, es.ask)  # see method __call__
    ...     es.countevals += nh.evaluations_just_done  # this is a hack, not important though
    ...     logger.add(more_data = [nh.evaluations, nh.noiseS])  # add a data point
    ...     es.disp()
    ...     # nh.maxevals = ...  it might be useful to start with smaller values and then increase
    >>> print(es.stop())
    >>> print(es.result()[-2])  # take mean value, the best solution is totally off
    >>> assert sum(es.result()[-2]**2) < 1e-9
    >>> print(X[np.argmin(fit)])  # not bad, but probably worse than the mean
    >>> logger.plot()

    The noise options of `fmin()` control a `NoiseHandler` instance similar to this
    example. The command ``cma.Options('noise')`` lists in effect the parameters of
    `__init__` apart from ``aggregate``.

    Details
    -------
    The parameters reevals, theta, c_s, and alpha_t are set differently
    than in the original publication, see method `__init__()`. For a
    very small population size, say popsize <= 5, the measurement
    technique based on rank changes is likely to fail.

    Missing Features
    ----------------
    In case no noise is found, ``self.lam_reeval`` should be adaptive
    and get at least as low as 1 (however the possible savings from this
    are rather limited). Another option might be to decide during the
    first call by a quantitative analysis of fitness values whether
    ``lam_reeval`` is set to zero. More generally, an automatic noise
    mode detection might also set the covariance matrix learning rates
    to smaller values.

    :See: `fmin()`, `ask_and_eval()`

    """
    def __init__(self, N, maxevals=10, aggregate=np.median, reevals=None, epsilon=1e-7, parallel=False):
        """parameters are
            `N`
                dimension
            `maxevals`
                maximal value for ``self.evaluations``, where
                ``self.evaluations`` function calls are aggregated for
                noise treatment. With ``maxevals == 0`` the noise
                handler is (temporarily) "switched off". If `maxevals`
                is a list, min value and (for >2 elements) median are
                used to define minimal and initial value of
                ``self.evaluations``. Choosing ``maxevals > 1`` is only
                reasonable, if also the original ``fit`` values (that
                are passed to `__call__`) are computed by aggregation of
                ``self.evaluations`` values (otherwise the values are
                not comparable), as it is done within `fmin()`.
            `aggregate`
                function to aggregate single f-values to a 'fitness', e.g.
                ``np.median``.
            `reevals`
                number of solutions to be reevaluated for noise measurement,
                can be a float, by default set to ``1.5 + popsize/20``,
                zero switches noise handling off.
            `epsilon`
                multiplier for perturbation of the reevaluated solutions
            `parallel`
                a single f-call with all resampled solutions

            :See: `fmin()`, `Options`, `CMAEvolutionStrategy.ask_and_eval()`

        """
        self.lam_reeval = reevals  # 2 + popsize/20, see method indices(), originally 2 + popsize/10
        self.epsilon = epsilon
        self.parallel = parallel
        self.theta = 0.5  # originally 0.2
        self.cum = 0.3  # originally 1, 0.3 allows one disagreement of current point with resulting noiseS
        self.alphasigma = 1 + 2 / (N+10)
        self.alphaevals = 1 + 2 / (N+10)  # originally 1.5
        self.alphaevalsdown = self.alphaevals**-0.25  # originally 1/1.5
        self.evaluations = 1  # to aggregate for a single f-evaluation
        self.minevals = 1
        self.maxevals = int(np.max(maxevals))
        if hasattr(maxevals, '__contains__'):  # i.e. can deal with ``in``
            if len(maxevals) > 1:
                self.minevals = min(maxevals)
                self.evaluations = self.minevals
            if len(maxevals) > 2:
                self.evaluations = np.median(maxevals)
        self.f_aggregate = aggregate
        self.evaluations_just_done = 0  # actually conducted evals, only for documentation
        self.noiseS = 0

    def __call__(self, X, fit, func, ask=None, args=()):
        """proceed with noise measurement, set anew attributes ``evaluations``
        (proposed number of evaluations to "treat" noise) and ``evaluations_just_done``
        and return a factor for increasing sigma.

        Parameters
        ----------
            `X`
                a list/sequence/vector of solutions
            `fit`
                the respective list of function values
            `func`
                the objective function, ``fit[i]`` corresponds to ``func(X[i], *args)``
            `ask`
                a method to generate a new, slightly disturbed solution. The argument
                is mandatory if ``epsilon`` is not zero, see `__init__()`.
            `args`
                optional additional arguments to `func`

        Details
        -------
        Calls the methods ``reeval()``, ``update_measure()`` and ``treat()`` in this order.
        ``self.evaluations`` is adapted within the method `treat()`.

        """
        self.evaluations_just_done = 0
        if not self.maxevals or self.lam_reeval == 0:
            return 1.0
        res = self.reeval(X, fit, func, ask, args)
        if not len(res):
            return 1.0
        self.update_measure()
        return self.treat()

    def get_evaluations(self):
        """return ``self.evaluations``, the number of evalutions to get a single fitness measurement"""
        return self.evaluations

    def treat(self):
        """adapt self.evaluations depending on the current measurement value
        and return ``sigma_fac in (1.0, self.alphasigma)``

        """
        if self.noiseS > 0:
            self.evaluations = min((self.evaluations * self.alphaevals, self.maxevals))
            return self.alphasigma
        else:
            self.evaluations = max((self.evaluations * self.alphaevalsdown, self.minevals))
            return 1.0

    def reeval(self, X, fit, func, ask, args=()):
        """store two fitness lists, `fit` and ``fitre`` reevaluating some
        solutions in `X`.
        ``self.evaluations`` evaluations are done for each reevaluated
        fitness value.
        See `__call__()`, where `reeval()` is called.

        """
        self.fit = list(fit)
        self.fitre = list(fit)
        self.idx = self.indices(fit)
        if not len(self.idx):
            return self.idx
        evals = int(self.evaluations) if self.f_aggregate else 1
        fagg = np.median if self.f_aggregate is None else self.f_aggregate
        for i in self.idx:
            if self.epsilon:
                if self.parallel:
                    self.fitre[i] = fagg(func(ask(evals, X[i], self.epsilon), *args))
                else:
                    self.fitre[i] = fagg([func(ask(1, X[i], self.epsilon)[0], *args)
                                            for _k in xrange(evals)])
            else:
                self.fitre[i] = fagg([func(X[i], *args) for _k in xrange(evals)])
        self.evaluations_just_done = evals * len(self.idx)
        return self.fit, self.fitre, self.idx

    def update_measure(self):
        """updated noise level measure using two fitness lists ``self.fit`` and
        ``self.fitre``, return ``self.noiseS, all_individual_measures``.

        Assumes that `self.idx` contains the indices where the fitness
        lists differ

        """
        lam = len(self.fit)
        idx = np.argsort(self.fit + self.fitre)
        ranks = np.argsort(idx).reshape((2, lam))
        rankDelta = ranks[0] - ranks[1] - np.sign(ranks[0] - ranks[1])

        # compute rank change limits using both ranks[0] and ranks[1]
        r = np.arange(1, 2 * lam)  # 2 * lam - 2 elements
        limits = [0.5 * (Mh.prctile(np.abs(r - (ranks[0,i] + 1 - (ranks[0,i] > ranks[1,i]))),
                                      self.theta*50) +
                         Mh.prctile(np.abs(r - (ranks[1,i] + 1 - (ranks[1,i] > ranks[0,i]))),
                                      self.theta*50))
                    for i in self.idx]
        # compute measurement
        #                               max: 1 rankchange in 2*lambda is always fine
        s = np.abs(rankDelta[self.idx]) - Mh.amax(limits, 1)  # lives roughly in 0..2*lambda
        self.noiseS += self.cum * (np.mean(s) - self.noiseS)
        return self.noiseS, s

    def indices(self, fit):
        """return the set of indices to be reevaluted for noise measurement,
        taking the ``lam_reeval`` best from the first ``2 * lam_reeval + 2``
        values.

        Given the first values are the earliest, this is a useful policy also
        with a time changing objective.

        """
        lam = self.lam_reeval if self.lam_reeval else 2 + len(fit) / 20
        reev = int(lam) + ((lam % 1) > np.random.rand())
        return np.argsort(array(fit, copy=False)[:2 * (reev + 1)])[:reev]

#____________________________________________________________
#____________________________________________________________
class Sections(object):
    """plot sections through an objective function. A first
    rational thing to do, when facing an (expensive) application.
    By default 6 points in each coordinate are evaluated.
    This class is still experimental.

    Examples
    --------

    >>> import cma, numpy as np
    >>> s = cma.Sections(cma.Fcts.rosen, np.zeros(3)).do(plot=False)
    >>> s.do(plot=False)  # evaluate the same points again, i.e. check for noise
    >>> try:
    ...     s.plot()
    ... except:
    ...     print('plotting failed: pylab package is missing?')

    Details
    -------
    Data are saved after each function call during `do()`. The filename is attribute
    ``name`` and by default ``str(func)``, see `__init__()`.

    A random (orthogonal) basis can be generated with ``cma.Rotation()(np.eye(3))``.

    The default name is unique in the function name, but it should be unique in all
    parameters of `__init__()` but `plot_cmd` and `load`.

    ``self.res`` is a dictionary with an entry for each "coordinate" ``i`` and with an
    entry ``'x'``, the middle point. Each entry ``i`` is again a dictionary with keys
    being different dx values and the value being a sequence of f-values.
    For example ``self.res[2][0.1] == [0.01, 0.01]``, which is generated using the
    difference vector ``self.basis[2]`` like
    ``self.res[2][dx] += func(self.res['x'] + dx * self.basis[2])``.

    :See: `__init__()`

    """
    def __init__(self, func, x, args=(), basis=None, name=None,
                 plot_cmd=pylab.plot if pylab else None, load=True):
        """
        Parameters
        ----------
            `func`
                objective function
            `x`
                point in search space, middle point of the sections
            `args`
                arguments passed to `func`
            `basis`
                evaluated points are ``func(x + locations[j] * basis[i]) for i in len(basis) for j in len(locations)``,
                see `do()`
            `name`
                filename where to save the result
            `plot_cmd`
                command used to plot the data, typically matplotlib pylabs `plot` or `semilogy`
            `load`
                load previous data from file ``str(func) + '.pkl'``

        """
        self.func = func
        self.args = args
        self.x = x
        self.name = name if name else str(func).replace(' ', '_').replace('>', '').replace('<', '')
        self.plot_cmd = plot_cmd  # or semilogy
        self.basis = np.eye(len(x)) if basis is None else basis

        try:
            self.load()
            if any(self.res['x'] != x):
                self.res = {}
                self.res['x'] = x  # TODO: res['x'] does not look perfect
            else:
                print(self.name + ' loaded')
        except:
            self.res = {}
            self.res['x'] = x

    def do(self, repetitions=1, locations=np.arange(-0.5, 0.6, 0.2), plot=True):
        """generates, plots and saves function values ``func(y)``,
        where ``y`` is 'close' to `x` (see `__init__()`). The data are stored in
        the ``res`` attribute and the class instance is saved in a file
        with (the weired) name ``str(func)``.

        Parameters
        ----------
            `repetitions`
                for each point, only for noisy functions is >1 useful. For
                ``repetitions==0`` only already generated data are plotted.
            `locations`
                coordinated wise deviations from the middle point given in `__init__`

        """
        if not repetitions:
            self.plot()
            return

        res = self.res
        for i in range(len(self.basis)): # i-th coordinate
            if i not in res:
                res[i] = {}
            # xx = np.array(self.x)
            # TODO: store res[i]['dx'] = self.basis[i] here?
            for dx in locations:
                xx = self.x + dx * self.basis[i]
                xkey = dx  # xx[i] if (self.basis == np.eye(len(self.basis))).all() else dx
                if xkey not in res[i]:
                    res[i][xkey] = []
                n = repetitions
                while n > 0:
                    n -= 1
                    res[i][xkey].append(self.func(xx, *self.args))
                    if plot:
                        self.plot()
                    self.save()
        return self

    def plot(self, plot_cmd=None, tf=lambda y: y):
        """plot the data we have, return ``self``"""
        if not plot_cmd:
            plot_cmd = self.plot_cmd
        colors = 'bgrcmyk'
        pylab.hold(False)
        res = self.res

        flatx, flatf = self.flattened()
        minf = np.inf
        for i in flatf:
            minf = min((minf, min(flatf[i])))
        addf = 1e-9 - minf  if minf <= 0 else 0
        for i in sorted(res.keys()):  # we plot not all values here
            if type(i) is int:
                color = colors[i % len(colors)]
                arx = sorted(res[i].keys())
                plot_cmd(arx, [tf(np.median(res[i][x]) + addf) for x in arx], color + '-')
                pylab.text(arx[-1], tf(np.median(res[i][arx[-1]])), i)
                pylab.hold(True)
                plot_cmd(flatx[i], tf(np.array(flatf[i]) + addf), color + 'o')
        pylab.ylabel('f + ' + str(addf))
        pylab.draw()
        show()
        # raw_input('press return')
        return self

    def flattened(self):
        """return flattened data ``(x, f)`` such that for the sweep through
        coordinate ``i`` we have for data point ``j`` that ``f[i][j] == func(x[i][j])``

        """
        flatx = {}
        flatf = {}
        for i in self.res:
            if type(i) is int:
                flatx[i] = []
                flatf[i] = []
                for x in sorted(self.res[i]):
                    for d in sorted(self.res[i][x]):
                        flatx[i].append(x)
                        flatf[i].append(d)
        return flatx, flatf

    def save(self, name=None):
        """save to file"""
        import pickle
        name = name if name else self.name
        fun = self.func
        del self.func  # instance method produces error
        pickle.dump(self, open(name + '.pkl', "wb" ))
        self.func = fun
        return self

    def load(self, name=None):
        """load from file"""
        import pickle
        name = name if name else self.name
        s = pickle.load(open(name + '.pkl', 'rb'))
        self.res = s.res  # disregard the class
        return self
#____________________________________________________________
#____________________________________________________________
class _Error(Exception):
    """generic exception of cma module"""
    pass

#____________________________________________________________
#____________________________________________________________
#
class ElapsedTime(object):
    """32-bit C overflows after int(2**32/1e6) == 4294s about 72 min"""
    def __init__(self):
        self.tic0 = time.clock()
        self.tic = self.tic0
        self.lasttoc = time.clock()
        self.lastdiff = time.clock() - self.lasttoc
        self.time_to_add = 0
        self.messages = 0

    def __call__(self):
        toc = time.clock()
        if toc - self.tic >= self.lasttoc - self.tic:
            self.lastdiff = toc - self.lasttoc
            self.lasttoc = toc
        else:  # overflow, reset self.tic
            if self.messages < 3:
                self.messages += 1
                print('  in cma.ElapsedTime: time measure overflow, last difference estimated from',
                        self.tic0, self.tic, self.lasttoc, toc, toc - self.lasttoc, self.lastdiff)

            self.time_to_add += self.lastdiff + self.lasttoc - self.tic
            self.tic = toc  # reset
            self.lasttoc = toc
        self.elapsedtime = toc - self.tic + self.time_to_add
        return self.elapsedtime

#____________________________________________________________
#____________________________________________________________
#
class TimeIt(object):
    def __init__(self, fct, args=(), seconds=1):
        pass

class Misc(object):
    #____________________________________________________________
    #____________________________________________________________
    #
    class MathHelperFunctions(object):
        """static convenience math helper functions, if the function name
        is preceded with an "a", a numpy array is returned

        """
        @staticmethod
        def aclamp(x, upper):
            return -Misc.MathHelperFunctions.apos(-x, -upper)
        @staticmethod
        def expms(A, eig=np.linalg.eigh):
            """matrix exponential for a symmetric matrix"""
            # TODO: check that this works reliably for low rank matrices
            # first: symmetrize A
            D, B = eig(A)
            return np.dot(B, (np.exp(D) * B).T)
        @staticmethod
        def amax(vec, vec_or_scalar):
            return array(Misc.MathHelperFunctions.max(vec, vec_or_scalar))
        @staticmethod
        def max(vec, vec_or_scalar):
            b = vec_or_scalar
            if np.isscalar(b):
                m = [max(x, b) for x in vec]
            else:
                m = [max(vec[i], b[i]) for i in xrange(len(vec))]
            return m
        @staticmethod
        def amin(vec_or_scalar, vec_or_scalar2):
            return array(Misc.MathHelperFunctions.min(vec_or_scalar, vec_or_scalar2))
        @staticmethod
        def min(a, b):
            iss = np.isscalar
            if iss(a) and iss(b):
                return min(a, b)
            if iss(a):
                a, b = b, a
            # now only b can be still a scalar
            if iss(b):
                return [min(x, b) for x in a]
            else:  # two non-scalars must have the same length
                return [min(a[i], b[i]) for i in xrange(len(a))]
        @staticmethod
        def norm(vec, expo=2):
            return sum(vec**expo)**(1/expo)
        @staticmethod
        def apos(x, lower=0):
            """clips argument (scalar or array) from below at lower"""
            if lower == 0:
                return (x > 0) * x
            else:
                return lower + (x > lower) * (x - lower)
        @staticmethod
        def prctile(data, p_vals=[0, 25, 50, 75, 100], sorted_=False):
            """``prctile(data, 50)`` returns the median, but p_vals can
            also be a sequence.

            Provides for small samples better values than matplotlib.mlab.prctile,
            however also slower.

            """
            ps = [p_vals] if np.isscalar(p_vals) else p_vals

            if not sorted_:
                data = sorted(data)
            n = len(data)
            d = []
            for p in ps:
                fi = p * n / 100 - 0.5
                if fi <= 0:  # maybe extrapolate?
                    d.append(data[0])
                elif fi >= n - 1:
                    d.append(data[-1])
                else:
                    i = int(fi)
                    d.append((i+1 - fi) * data[i] + (fi - i) * data[i+1])
            return d[0] if np.isscalar(p_vals) else d
        @staticmethod
        def sround(nb):  # TODO: to be vectorized
            """return stochastic round: floor(nb) + (rand()<remainder(nb))"""
            return nb // 1 + (np.random.rand(1)[0] < (nb % 1))

        @staticmethod
        def cauchy_with_variance_one():
            n = np.random.randn() / np.random.randn()
            while abs(n) > 1000:
                n = np.random.randn() / np.random.randn()
            return n / 25
        @staticmethod
        def standard_finite_cauchy(size=1):
            try:
                l = len(size)
            except TypeError:
                l = 0

            if l == 0:
                return array([Mh.cauchy_with_variance_one() for _i in xrange(size)])
            elif l == 1:
                return array([Mh.cauchy_with_variance_one() for _i in xrange(size[0])])
            elif l == 2:
                return array([[Mh.cauchy_with_variance_one() for _i in xrange(size[1])]
                             for _j in xrange(size[0])])
            else:
                raise _Error('len(size) cannot be large than two')


    @staticmethod
    def likelihood(x, m=None, Cinv=None, sigma=1, detC=None):
        """return likelihood of x for the normal density N(m, sigma**2 * Cinv**-1)"""
        # testing: MC integrate must be one: mean(p(x_i)) * volume(where x_i are uniformely sampled)
        # for i in range(3): print mean([cma.likelihood(20*r-10, dim * [0], None, 3) for r in rand(10000,dim)]) * 20**dim
        if m is None:
            dx = x
        else:
            dx = x - m  # array(x) - array(m)
        n = len(x)
        s2pi = (2*np.pi)**(n/2.)
        if Cinv is None:
            return exp(-sum(dx**2) / sigma**2 / 2) / s2pi / sigma**n
        if detC is None:
            detC = 1. / np.linalg.linalg.det(Cinv)
        return  exp(-np.dot(dx, np.dot(Cinv, dx)) / sigma**2 / 2) / s2pi / abs(detC)**0.5 / sigma**n

    @staticmethod
    def loglikelihood(self, x, previous=False):
        """return log-likelihood of `x` regarding the current sample distribution"""
        # testing of original fct: MC integrate must be one: mean(p(x_i)) * volume(where x_i are uniformely sampled)
        # for i in range(3): print mean([cma.likelihood(20*r-10, dim * [0], None, 3) for r in rand(10000,dim)]) * 20**dim
        # TODO: test this!!
        # c=cma.fmin...
        # c[3]['cma'].loglikelihood(...)

        if previous and hasattr(self, 'lastiter'):
            sigma = self.lastiter.sigma
            Crootinv = self.lastiter._Crootinv
            xmean = self.lastiter.mean
            D = self.lastiter.D
        elif previous and self.countiter > 1:
            raise _Error('no previous distribution parameters stored, check options importance_mixing')
        else:
            sigma = self.sigma
            Crootinv = self._Crootinv
            xmean = self.mean
            D = self.D

        dx = array(x) - xmean  # array(x) - array(m)
        n = self.N
        logs2pi = n * log(2*np.pi) / 2.
        logdetC = 2 * sum(log(D))
        dx = np.dot(Crootinv, dx)
        res = -sum(dx**2) / sigma**2 / 2 - logs2pi - logdetC/2 - n*log(sigma)
        if 1 < 3: # testing
            s2pi = (2*np.pi)**(n/2.)
            detC = np.prod(D)**2
            res2 = -sum(dx**2) / sigma**2 / 2 - log(s2pi * abs(detC)**0.5 * sigma**n)
            assert res2 < res + 1e-8 or res2 > res - 1e-8
        return res

    #____________________________________________________________
    #____________________________________________________________
    #
    # C and B are arrays rather than matrices, because they are
    # addressed via B[i][j], matrices can only be addressed via B[i,j]

    # tred2(N, B, diagD, offdiag);
    # tql2(N, diagD, offdiag, B);


    # Symmetric Householder reduction to tridiagonal form, translated from JAMA package.
    @staticmethod
    def eig(C):
        """eigendecomposition of a symmetric matrix, much slower than
        `numpy.linalg.eigh`, return ``(EVals, Basis)``, the eigenvalues
        and an orthonormal basis of the corresponding eigenvectors, where

            ``Basis[i]``
                the i-th row of ``Basis``
            columns of ``Basis``, ``[Basis[j][i] for j in range(len(Basis))]``
                the i-th eigenvector with eigenvalue ``EVals[i]``

        """

    # class eig(object):
    #     def __call__(self, C):

    # Householder transformation of a symmetric matrix V into tridiagonal form.
        # -> n             : dimension
        # -> V             : symmetric nxn-matrix
        # <- V             : orthogonal transformation matrix:
        #                    tridiag matrix == V * V_in * V^t
        # <- d             : diagonal
        # <- e[0..n-1]     : off diagonal (elements 1..n-1)

        # Symmetric tridiagonal QL algorithm, iterative
        # Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3 operations
        # -> n     : Dimension.
        # -> d     : Diagonale of tridiagonal matrix.
        # -> e[1..n-1] : off-diagonal, output from Householder
        # -> V     : matrix output von Householder
        # <- d     : eigenvalues
        # <- e     : garbage?
        # <- V     : basis of eigenvectors, according to d


        #  tred2(N, B, diagD, offdiag); B=C on input
        #  tql2(N, diagD, offdiag, B);

        #  private void tred2 (int n, double V[][], double d[], double e[]) {
        def tred2 (n, V, d, e):
            #  This is derived from the Algol procedures tred2 by
            #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            #  Fortran subroutine in EISPACK.

            num_opt = False  # factor 1.5 in 30-D

            for j in range(n):
                d[j] = V[n-1][j] # d is output argument

            # Householder reduction to tridiagonal form.

            for i in range(n-1,0,-1):
                # Scale to avoid under/overflow.
                h = 0.0
                if not num_opt:
                    scale = 0.0
                    for k in range(i):
                        scale = scale + abs(d[k])
                else:
                    scale = sum(abs(d[0:i]))

                if scale == 0.0:
                    e[i] = d[i-1]
                    for j in range(i):
                        d[j] = V[i-1][j]
                        V[i][j] = 0.0
                        V[j][i] = 0.0
                else:

                    # Generate Householder vector.
                    if not num_opt:
                        for k in range(i):
                            d[k] /= scale
                            h += d[k] * d[k]
                    else:
                        d[:i] /= scale
                        h = np.dot(d[:i],d[:i])

                    f = d[i-1]
                    g = h**0.5

                    if f > 0:
                        g = -g

                    e[i] = scale * g
                    h = h - f * g
                    d[i-1] = f - g
                    if not num_opt:
                        for j in range(i):
                            e[j] = 0.0
                    else:
                        e[:i] = 0.0

                    # Apply similarity transformation to remaining columns.

                    for j in range(i):
                        f = d[j]
                        V[j][i] = f
                        g = e[j] + V[j][j] * f
                        if not num_opt:
                            for k in range(j+1, i):
                                g += V[k][j] * d[k]
                                e[k] += V[k][j] * f
                            e[j] = g
                        else:
                            e[j+1:i] += V.T[j][j+1:i] * f
                            e[j] = g + np.dot(V.T[j][j+1:i],d[j+1:i])

                    f = 0.0
                    if not num_opt:
                        for j in range(i):
                            e[j] /= h
                            f += e[j] * d[j]
                    else:
                        e[:i] /= h
                        f += np.dot(e[:i],d[:i])

                    hh = f / (h + h)
                    if not num_opt:
                        for j in range(i):
                            e[j] -= hh * d[j]
                    else:
                        e[:i] -= hh * d[:i]

                    for j in range(i):
                        f = d[j]
                        g = e[j]
                        if not num_opt:
                            for k in range(j, i):
                                V[k][j] -= (f * e[k] + g * d[k])
                        else:
                            V.T[j][j:i] -= (f * e[j:i] + g * d[j:i])

                        d[j] = V[i-1][j]
                        V[i][j] = 0.0

                d[i] = h
            # end for i--

            # Accumulate transformations.

            for i in range(n-1):
                V[n-1][i] = V[i][i]
                V[i][i] = 1.0
                h = d[i+1]
                if h != 0.0:
                    if not num_opt:
                        for k in range(i+1):
                            d[k] = V[k][i+1] / h
                    else:
                        d[:i+1] = V.T[i+1][:i+1] / h

                    for j in range(i+1):
                        if not num_opt:
                            g = 0.0
                            for k in range(i+1):
                                g += V[k][i+1] * V[k][j]
                            for k in range(i+1):
                                V[k][j] -= g * d[k]
                        else:
                            g = np.dot(V.T[i+1][0:i+1], V.T[j][0:i+1])
                            V.T[j][:i+1] -= g * d[:i+1]

                if not num_opt:
                    for k in range(i+1):
                        V[k][i+1] = 0.0
                else:
                    V.T[i+1][:i+1] = 0.0


            if not num_opt:
                for j in range(n):
                    d[j] = V[n-1][j]
                    V[n-1][j] = 0.0
            else:
                d[:n] = V[n-1][:n]
                V[n-1][:n] = 0.0

            V[n-1][n-1] = 1.0
            e[0] = 0.0


        # Symmetric tridiagonal QL algorithm, taken from JAMA package.
        # private void tql2 (int n, double d[], double e[], double V[][]) {
        # needs roughly 3N^3 operations
        def tql2 (n, d, e, V):

            #  This is derived from the Algol procedures tql2, by
            #  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            #  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            #  Fortran subroutine in EISPACK.

            num_opt = False  # using vectors from numpy makes it faster

            if not num_opt:
                for i in range(1,n): # (int i = 1; i < n; i++):
                    e[i-1] = e[i]
            else:
                e[0:n-1] = e[1:n]
            e[n-1] = 0.0

            f = 0.0
            tst1 = 0.0
            eps = 2.0**-52.0
            for l in range(n): # (int l = 0; l < n; l++) {

                # Find small subdiagonal element

                tst1 = max(tst1, abs(d[l]) + abs(e[l]))
                m = l
                while m < n:
                    if abs(e[m]) <= eps*tst1:
                        break
                    m += 1

                # If m == l, d[l] is an eigenvalue,
                # otherwise, iterate.

                if m > l:
                    iiter = 0
                    while 1: # do {
                        iiter += 1  # (Could check iteration count here.)

                        # Compute implicit shift

                        g = d[l]
                        p = (d[l+1] - g) / (2.0 * e[l])
                        r = (p**2 + 1)**0.5  # hypot(p,1.0)
                        if p < 0:
                            r = -r

                        d[l] = e[l] / (p + r)
                        d[l+1] = e[l] * (p + r)
                        dl1 = d[l+1]
                        h = g - d[l]
                        if not num_opt:
                            for i in range(l+2, n):
                                d[i] -= h
                        else:
                            d[l+2:n] -= h

                        f = f + h

                        # Implicit QL transformation.

                        p = d[m]
                        c = 1.0
                        c2 = c
                        c3 = c
                        el1 = e[l+1]
                        s = 0.0
                        s2 = 0.0

                        # hh = V.T[0].copy()  # only with num_opt
                        for i in range(m-1, l-1, -1): # (int i = m-1; i >= l; i--) {
                            c3 = c2
                            c2 = c
                            s2 = s
                            g = c * e[i]
                            h = c * p
                            r = (p**2 + e[i]**2)**0.5  # hypot(p,e[i])
                            e[i+1] = s * r
                            s = e[i] / r
                            c = p / r
                            p = c * d[i] - s * g
                            d[i+1] = h + s * (c * g + s * d[i])

                            # Accumulate transformation.

                            if not num_opt: # overall factor 3 in 30-D
                                for k in range(n): # (int k = 0; k < n; k++) {
                                    h = V[k][i+1]
                                    V[k][i+1] = s * V[k][i] + c * h
                                    V[k][i] = c * V[k][i] - s * h
                            else: # about 20% faster in 10-D
                                hh = V.T[i+1].copy()
                                # hh[:] = V.T[i+1][:]
                                V.T[i+1] = s * V.T[i] + c * hh
                                V.T[i] = c * V.T[i] - s * hh
                                # V.T[i] *= c
                                # V.T[i] -= s * hh

                        p = -s * s2 * c3 * el1 * e[l] / dl1
                        e[l] = s * p
                        d[l] = c * p

                        # Check for convergence.
                        if abs(e[l]) <= eps*tst1:
                            break
                    # } while (Math.abs(e[l]) > eps*tst1);

                d[l] = d[l] + f
                e[l] = 0.0


            # Sort eigenvalues and corresponding vectors.
            if 11 < 3:
                for i in range(n-1): # (int i = 0; i < n-1; i++) {
                    k = i
                    p = d[i]
                    for j in range(i+1, n): # (int j = i+1; j < n; j++) {
                        if d[j] < p: # NH find smallest k>i
                            k = j
                            p = d[j]

                    if k != i:
                        d[k] = d[i] # swap k and i
                        d[i] = p
                        for j in range(n): # (int j = 0; j < n; j++) {
                            p = V[j][i]
                            V[j][i] = V[j][k]
                            V[j][k] = p
        # tql2

        N = len(C[0])
        if 11 < 3:
            V = np.array([x[:] for x in C])  # copy each "row"
            N = V[0].size
            d = np.zeros(N)
            e = np.zeros(N)
        else:
            V = [[x[i] for i in xrange(N)] for x in C]  # copy each "row"
            d = N * [0.]
            e = N * [0.]

        tred2(N, V, d, e)
        tql2(N, d, e, V)
        return (array(d), array(V))
Mh = Misc.MathHelperFunctions
def pprint(to_be_printed):
    """nicely formated print"""
    try:
        import pprint as pp
        # generate an instance PrettyPrinter
        # pp.PrettyPrinter().pprint(to_be_printed)
        pp.pprint(to_be_printed)
    except ImportError:
        print('could not use pprint module, will apply regular print')
        print(to_be_printed)
class Rotation(object):
    """Rotation class that implements an orthogonal linear transformation,
    one for each dimension. Used to implement non-separable test functions.

    Example:

    >>> import cma, numpy as np
    >>> R = cma.Rotation()
    >>> R2 = cma.Rotation() # another rotation
    >>> x = np.array((1,2,3))
    >>> print(R(R(x), inverse=1))
    [ 1.  2.  3.]

    """
    dicMatrices = {}  # store matrix if necessary, for each dimension
    def __init__(self):
        self.dicMatrices = {} # otherwise there might be shared bases which is probably not what we want
    def __call__(self, x, inverse=False): # function when calling an object
        """Rotates the input array `x` with a fixed rotation matrix
           (``self.dicMatrices['str(len(x))']``)
        """
        N = x.shape[0]  # can be an array or matrix, TODO: accept also a list of arrays?
        if str(N) not in self.dicMatrices: # create new N-basis for once and all
            B = np.random.randn(N, N)
            for i in xrange(N):
                for j in xrange(0, i):
                    B[i] -= np.dot(B[i], B[j]) * B[j]
                B[i] /= sum(B[i]**2)**0.5
            self.dicMatrices[str(N)] = B
        if inverse:
            return np.dot(self.dicMatrices[str(N)].T, x)  # compute rotation
        else:
            return np.dot(self.dicMatrices[str(N)], x)  # compute rotation
# Use rotate(x) to rotate x
rotate = Rotation()

#____________________________________________________________
#____________________________________________________________
#
class FitnessFunctions(object):
    """ versatile container for test objective functions """

    def __init__(self):
        self.counter = 0  # number of calls or any other practical use
    def rot(self, x, fun, rot=1, args=()):
        """returns ``fun(rotation(x), *args)``, ie. `fun` applied to a rotated argument"""
        if len(np.shape(array(x))) > 1:  # parallelized
            res = []
            for x in x:
                res.append(self.rot(x, fun, rot, args))
            return res

        if rot:
            return fun(rotate(x, *args))
        else:
            return fun(x)
    def somenan(self, x, fun, p=0.1):
        """returns sometimes np.NaN, otherwise fun(x)"""
        if np.random.rand(1) < p:
            return np.NaN
        else:
            return fun(x)
    def rand(self, x):
        """Random test objective function"""
        return np.random.random(1)[0]
    def linear(self, x):
        return -x[0]
    def lineard(self, x):
        if 1 < 3 and any(array(x) < 0):
            return np.nan
        if 1 < 3 and sum([ (10 + i) * x[i] for i in xrange(len(x))]) > 50e3:
            return np.nan
        return -sum(x)
    def sphere(self, x):
        """Sphere (squared norm) test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        return sum((x+0)**2)
    def spherewithoneconstraint(self, x):
        return sum((x+0)**2) if x[0] > 1 else np.nan
    def elliwithoneconstraint(self, x, idx=[-1]):
        return self.ellirot(x) if all(array(x)[idx] > 1) else np.nan

    def spherewithnconstraints(self, x):
        return sum((x+0)**2) if all(array(x) > 1) else np.nan

    def noisysphere(self, x, noise=4.0, cond=1.0):
        """noise=10 does not work with default popsize, noise handling does not help """
        return self.elli(x, cond=cond) * (1 + noise * np.random.randn() / len(x))
    def spherew(self, x):
        """Sphere (squared norm) with sum x_i = 1 test objective function"""
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        # s = sum(abs(x))
        # return sum((x/s+0)**2) - 1/len(x)
        # return sum((x/s)**2) - 1/len(x)
        return -0.01*x[0] + abs(x[0])**-2 * sum(x[1:]**2)
    def partsphere(self, x):
        """Sphere (squared norm) test objective function"""
        self.counter += 1
        # return np.random.rand(1)[0]**0 * sum(x**2) + 1 * np.random.rand(1)[0]
        dim = len(x)
        x = array([x[i % dim] for i in range(2*dim)])
        N = 8
        i = self.counter % dim
        #f = sum(x[i:i + N]**2)
        f = sum(x[np.random.randint(dim, size=N)]**2)
        return f
    def sectorsphere(self, x):
        """asymmetric Sphere (squared norm) test objective function"""
        return sum(x**2) + (1e6-1) * sum(x[x<0]**2)
    def cornersphere(self, x):
        """Sphere (squared norm) test objective function constraint to the corner"""
        nconstr = len(x) - 0
        if any(x[:nconstr] < 1):
            return np.NaN
        return sum(x**2) - nconstr
    def cornerelli(self, x):
        """ """
        if any(x < 1):
            return np.NaN
        return self.elli(x) - self.elli(np.ones(len(x)))
    def cornerellirot(self, x):
        """ """
        if any(x < 1):
            return np.NaN
        return self.ellirot(x)
    def normalSkew(self, f):
        N = np.random.randn(1)[0]**2
        if N < 1:
            N = f * N  # diminish blow up lower part
        return N
    def noiseC(self, x, func=sphere, fac=10, expon=0.8):
        f = func(self, x)
        N = np.random.randn(1)[0]/np.random.randn(1)[0]
        return max(1e-19, f + (float(fac)/len(x)) * f**expon * N)
    def noise(self, x, func=sphere, fac=10, expon=1):
        f = func(self, x)
        #R = np.random.randn(1)[0]
        R = np.log10(f) + expon * abs(10-np.log10(f)) * np.random.rand(1)[0]
        # sig = float(fac)/float(len(x))
        # R = log(f) + 0.5*log(f) * random.randn(1)[0]
        # return max(1e-19, f + sig * (f**np.log10(f)) * np.exp(R))
        # return max(1e-19, f * np.exp(sig * N / f**expon))
        # return max(1e-19, f * normalSkew(f**expon)**sig)
        return f + 10**R  # == f + f**(1+0.5*RN)
    def cigar(self, x, rot=0, cond=1e6):
        """Cigar test objective function"""
        if rot:
            x = rotate(x)
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [x[0]**2 + cond * sum(x[1:]**2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def tablet(self, x, rot=0):
        """Tablet test objective function"""
        if rot:
            x = rotate(x)
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [1e6*x[0]**2 + sum(x[1:]**2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def cigtab(self, y):
        """Cigtab test objective function"""
        X = [y] if np.isscalar(y[0]) else y
        f = [1e-4 * x[0]**2 + 1e4 * x[1]**2 + sum(x[2:]**2) for x in X]
        return f if len(f) > 1 else f[0]
    def twoaxes(self, y):
        """Cigtab test objective function"""
        X = [y] if np.isscalar(y[0]) else y
        N2 = len(X[0]) // 2
        f = [1e6 * sum(x[0:N2]**2) + sum(x[N2:]**2) for x in X]
        return f if len(f) > 1 else f[0]
    def ellirot(self, x):
        return fcts.elli(array(x), 1)
    def hyperelli(self, x):
        N = len(x)
        return sum((np.arange(1, N+1) * x)**2)
    def elli(self, x, rot=0, xoffset=0, cond=1e6, actuator_noise=0.0, both=False):
        """Ellipsoid test objective function"""
        if not np.isscalar(x[0]):  # parallel evaluation
            return [self.elli(xi, rot) for xi in x]  # could save 20% overall
        if rot:
            x = rotate(x)
        N = len(x)
        if actuator_noise:
            x = x + actuator_noise * np.random.randn(N)

        ftrue = sum(cond**(np.arange(N)/(N-1.))*(x+xoffset)**2)

        alpha = 0.49 + 1./N
        beta = 1
        felli = np.random.rand(1)[0]**beta * ftrue * \
                max(1, (10.**9 / (ftrue+1e-99))**(alpha*np.random.rand(1)[0]))
        # felli = ftrue + 1*np.random.randn(1)[0] / (1e-30 +
        #                                           np.abs(np.random.randn(1)[0]))**0
        if both:
            return (felli, ftrue)
        else:
            # return felli  # possibly noisy value
            return ftrue # + np.random.randn()
    def elliconstraint(self, x, cfac = 1e8, tough=True, cond=1e6):
        """ellipsoid test objective function with "constraints" """
        N = len(x)
        f = sum(cond**(np.arange(N)[-1::-1]/(N-1)) * x**2)
        cvals = (x[0] + 1,
                 x[0] + 1 + 100*x[1],
                 x[0] + 1 - 100*x[1])
        if tough:
            f += cfac * sum(max(0,c) for c in cvals)
        else:
            f += cfac * sum(max(0,c+1e-3)**2 for c in cvals)
        return f
    def rosen(self, x, alpha=1e2):
        """Rosenbrock test objective function"""
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [sum(alpha*(x[:-1]**2-x[1:])**2 + (1.-x[:-1])**2) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def diffpow(self, x, rot=0):
        """Diffpow test objective function"""
        N = len(x)
        if rot:
            x = rotate(x)
        return sum(np.abs(x)**(2.+4.*np.arange(N)/(N-1.)))**0.5
    def rosenelli(self, x):
        N = len(x)
        return self.rosen(x[:N/2]) + self.elli(x[N/2:], cond=1)
    def ridge(self, x, expo=2):
        x = [x] if np.isscalar(x[0]) else x  # scalar into list
        f = [x[0] + 100*np.sum(x[1:]**2)**(expo/2.) for x in x]
        return f if len(f) > 1 else f[0]  # 1-element-list into scalar
    def ridgecircle(self, x, expo=0.5):
        """happy cat by HG Beyer"""
        a = len(x)
        s = sum(x**2)
        return ((s - a)**2)**(expo/2) + s/a + sum(x)/a
    def happycat(self, x, alpha=1./8):
        s = sum(x**2)
        return ((s - len(x))**2)**alpha + (s/2 + sum(x)) / len(x) + 0.5
    def flat(self,x):
        return 1
        return 1 if np.random.rand(1) < 0.9 else 1.1
        return np.random.randint(1,30)
    def branin(self, x):
        # in [0,15]**2
        y = x[1]
        x = x[0] + 5
        return (y - 5.1*x**2 / 4 / np.pi**2 + 5 * x / np.pi - 6)**2 + 10 * (1 - 1/8/np.pi) * np.cos(x) + 10 - 0.397887357729738160000
    def goldsteinprice(self, x):
        x1 = x[0]
        x2 = x[1]
        return (1 + (x1 +x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)) * (
                30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)) - 3
    def griewank(self, x):
        # was in [-600 600]
        x = (600./5) * x
        return 1 - np.prod(np.cos(x/sqrt(1.+np.arange(len(x))))) + sum(x**2)/4e3
    def rastrigin(self, x):
        """Rastrigin test objective function"""
        if not np.isscalar(x[0]):
            N = len(x[0])
            return [10*N + sum(xi**2 - 10*np.cos(2*np.pi*xi)) for xi in x]
            # return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
        N = len(x)
        return 10*N + sum(x**2 - 10*np.cos(2*np.pi*x))
    def schaffer(self, x):
        """ Schaffer function x0 in [-100..100]"""
        N = len(x);
        s = x[0:N-1]**2 + x[1:N]**2;
        return sum(s**0.25 * (np.sin(50*s**0.1)**2 + 1))

    def schwefelelli(self, x):
        s = 0
        f = 0
        for i in xrange(len(x)):
            s += x[i]
            f += s**2
        return f
    def schwefelmult(self, x, pen_fac = 1e4):
        """multimodal Schwefel function with domain -500..500"""
        y = [x] if np.isscalar(x[0]) else x
        N = len(y[0])
        f = array([418.9829*N - 1.27275661e-5*N - sum(x * np.sin(np.abs(x)**0.5))
                + pen_fac * sum((abs(x) > 500) * (abs(x) - 500)**2) for x in y])
        return f if len(f) > 1 else f[0]
    def optprob(self, x):
        n = np.arange(len(x)) + 1
        f = n * x * (1-x)**(n-1)
        return sum(1-f)
    def lincon(self, x, theta=0.01):
        """ridge like linear function with one linear constraint"""
        if x[0] < 0:
            return np.NaN
        return theta * x[1] + x[0]
    def rosen_nesterov(self, x, rho=100):
        """needs exponential number of steps in a non-increasing f-sequence.

        x_0 = (-1,1,...,1)
        See Jarre (2011) "On Nesterov's Smooth Chebyshev-Rosenbrock Function"

        """
        f = 0.25 * (x[0] - 1)**2
        f += rho * sum((x[1:] - 2 * x[:-1]**2 + 1)**2)
        return f

fcts = FitnessFunctions()
Fcts = fcts  # for cross compatibility, as if the functions were static members of class Fcts
def felli(x): # unbound function, needed to test multiprocessor
    return sum(1e6**(np.arange(len(x))/(len(x)-1))*(x)**2)


#____________________________________________
#____________________________________________________________
def _test(module=None):  # None is fine when called from inside the module
    import doctest
    print(doctest.testmod(module))  # this is pretty coool!
def process_test(stream=None):
    """ """
    import fileinput
    s1 = ""
    s2 = ""
    s3 = ""
    state = 0
    for line in fileinput.input(stream):  # takes argv as file or stdin
        if 1 < 3:
            s3 += line
            if state < -1 and line.startswith('***'):
                print(s3)
            if line.startswith('***'):
                s3 = ""

        if state == -1:  # found a failed example line
            s1 += '\n\n*** Failed Example:' + line
            s2 += '\n\n\n'   # line
            # state = 0  # wait for 'Expected:' line

        if line.startswith('Expected:'):
            state = 1
            continue
        elif line.startswith('Got:'):
            state = 2
            continue
        elif line.startswith('***'):  # marks end of failed example
            state = 0
        elif line.startswith('Failed example:'):
            state = -1
        elif line.startswith('Exception raised'):
            state = -2

        # in effect more else:
        if state == 1:
            s1 += line + ''
        if state == 2:
            s2 += line + ''

#____________________________________________________________
#____________________________________________________________
#
def main(argv=None):
    """to install and/or test from the command line use::

        python cma.py [options | func dim sig0 [optkey optval][optkey optval]...]

    --test (or -t) to run the doctest, ``--test -v`` to get (much) verbosity
    and ``--test -q`` to run it quietly with output only in case of errors.

    install to install cma.py (uses setup from distutils.core).

    --fcts and --doc for more infos or start ipython --pylab.

    Examples
    --------
    First, testing with the local python distribution::

        python cma.py --test

    If succeeded install (uses setup from distutils.core)::

        python cma.py install

    A single run on the ellipsoid function::

        python cma.py elli 10 1

    """
    if argv is None:
        argv = sys.argv  # should have better been sys.argv[1:]

    # uncomment for unit test
    # _test()
    # handle input arguments, getopt might be helpful ;-)
    if len(argv) >= 1:  # function and help
        if len(argv) == 1 or argv[1].startswith('-h') or argv[1].startswith('--help'):
            print(main.__doc__)
            fun = None
        elif argv[1].startswith('-t') or argv[1].startswith('--test'):
            import doctest
            if len(argv) > 2 and (argv[2].startswith('--v') or argv[2].startswith('-v')):  # verbose
                print('doctest for cma.py: due to different platforms and python versions')
                print('and in some cases due to a missing unique random seed')
                print('many examples will "fail". This is OK, if they give a similar')
                print('to the expected result and if no exception occurs. ')
                # if argv[1][2] == 'v':
                doctest.testmod(report=True)  # this is quite cool!
            else:  # was: if len(argv) > 2 and (argv[2].startswith('--qu') or argv[2].startswith('-q')):
                print('doctest for cma.py: launching (it might be necessary to close a few pop up windows to finish)')
                fn = '__cma_doctest__.txt'
                stdout = sys.stdout
                try:
                    with open(fn, 'w') as f:
                        sys.stdout = f
                        doctest.testmod(report=True)  # this is quite cool!
                finally:
                    sys.stdout = stdout
                process_test(fn)
                print('doctest for cma.py: finished (no other output should be seen after launching)')
            return
        elif argv[1] == '--doc':
            print(__doc__)
            print(CMAEvolutionStrategy.__doc__)
            print(fmin.__doc__)
            fun = None
        elif argv[1] == '--fcts':
            print('List of valid function names:')
            print([d for d in dir(fcts) if not d.startswith('_')])
            fun = None
        elif argv[1] in ('install', '--install'):
            from distutils.core import setup
            setup(name = "cma",
                  version = __version__,
                  author = "Nikolaus Hansen",
                  #    packages = ["cma"],
                  py_modules = ["cma"],
                  )
            fun = None
        elif argv[1] in ('plot',):
            plot()
            raw_input('press return')
            fun = None
        elif len(argv) > 3:
            fun = eval('fcts.' + argv[1])
        else:
            print('try -h option')
            fun = None

    if fun is not None:

        if len(argv) > 2:  # dimension
            x0 = np.ones(eval(argv[2]))
        if len(argv) > 3:  # sigma
            sig0 = eval(argv[3])

        opts = {}
        for i in xrange(5, len(argv), 2):
            opts[argv[i-1]] = eval(argv[i])

        # run fmin
        if fun is not None:
            tic = time.time()
            fmin(fun, x0, sig0, **opts)  # ftarget=1e-9, tolfacupx=1e9, verb_log=10)
            # plot()
            # print ' best function value ', res[2]['es'].best[1]
            print('elapsed time [s]: + %.2f', round(time.time() - tic, 2))

    elif not len(argv):
        fmin(fcts.elli, np.ones(6)*0.1, 0.1, ftarget=1e-9)


#____________________________________________________________
#____________________________________________________________
#
# mainly for testing purpose
# executed when called from an OS shell
if __name__ == "__main__":
    # for i in range(1000):  # how to find the memory leak
    #     main(["cma.py", "rastrigin", "10", "5", "popsize", "200", "maxfevals", "24999", "verb_log", "0"])
    main()

