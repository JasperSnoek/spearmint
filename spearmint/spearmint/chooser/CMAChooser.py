from cma import CMAEvolutionStrategy
from spearmint import util
import Locker

def init(expt_dir, arg_string):
    args = util.unpack_args(arg_string)
    return CMAChooser(expt_dir, **args)

"""
Chooser module for the CMA-ES evolutionary optimizer.
"""
class CMAChooser:

    def __init__(self, expt_dir):

        raise NotImplementedError('The CMA chooser is not yet implemented!')
        
        self.state_pkl = os.path.join(expt_dir, self.__module__ + ".pkl")

        #TODO: params needs to be an array of starting values
        # - need to figure out how to map Spearmint params into
        # all floats usable by the evolution strategy.
        self.optimizer = CMAEvolutionStrategy(params)

    def _real_init(self, dims, values):

        raise NotImplementedError('The CMA chooser is not yet implemented!')
        self.locker.lock_wait(self.state_pkl)

        if os.path.exists(self.state_pkl):
            fh    = open(self.state_pkl, 'r')
            state = cPickle.load(fh)
            fh.close()

            #TODO: setup config and state values from state, or setup fresh
            #defaults

    def __del__(self):

        raise NotImplementedError('The CMA chooser is not yet implemented!')
        self.locker.lock_wait(self.state_pkl)

        # Write the hyperparameters out to a Pickle.
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)

        # do this to save the optimizer state
#    >>> pickle.dump(es, open('saved-cma-object.pkl', 'wb'))

        # and this to load it back...
#    >>> es = pickle.load(open('saved-cma-object.pkl', 'rb'))

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

    def next(self, grid, values, durations, candidates, pending, complete):

        raise NotImplementedError('The CMA chooser is not yet implemented!')

        # Perform the real initialization.
        if self.D == -1:
            self._real_init(grid.shape[1], values[complete])

        # Grab out the relevant sets.
        comp = grid[complete,:]
        cand = grid[candidates,:]
        pend = grid[pending,:]
        vals = values[complete]

        # TODO: tell the optimizer about any new f-values, get the next proposed
        # sample, or maybe generate a population of samples and iterate through
        # them?

#    ...         X = es.ask()    # get list of new solutions
#    ...         fit = [cma.fcts.rastrigin(x) for x in X]  # evaluate each solution
#    ...         es.tell(X, fit) # besides for termination only the ranking in fit is used
