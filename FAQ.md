Spearmint Frequently Asked Questions
------------------------------------

This document contains answers to some of the more frequent questions that spearmint users are asking.

**Question**: When I run spearmint my $PYTHONPATH variable appears to not be preserved. How do I fix this?  
**Answer**: If you invoke spearmint from the commandline using `bin/spearmint`, then spearmint is called from a bash script.  With bash scripts,
in order to preserve the environment you have to invoke them by "sourcing them" by putting a "." before the call `. bin/spearmint ...`
Another way to avoid this is running main.py directly (i.e. `python main.py ...`).  

**Question**: What exactly is passed into a Python wrapper by spearmint in the Params dictionary?  
**Answer**: For INT and FLOAT types spearmint passes in a numpy array of the size specified in config.pb in 'size' (even if it is size 1).  For ENUM types it is a list of strings of size 'size'.  

**Question**: Where can I see what my code outputs when it is being run by spearmint?  
**Answer**: In the experiment directory (i.e. the one that contains config.pb) spearmint adds an output directory that contains the output of each job that has been run.  The files are of the form <job_id>.out  

**Question**: Can I see a list of all the experiments run so far (i.e. a list of parameters) along with the corresponding function (cost or loss) values.  
**Answer**: The web status page (which you can invoke using the -w flag) now show this list in an easy to read table.  

**Question**: Quite often I would like to stop and restart spearmint. Sometimes I can Ctrl-C spearmint and restart it. Other times it seems to block and do nothing when restarted. Is there anything I can do except cleanup the directory and start from scratch again? Also Ctrl-C only seems to break me out of spearmint, a spearmint process is still running in the background (which I kill manually). Is this expected?  
**Answer**: Spearmint uses a file `expt_grid.pkl` to maintains state.  All the jobs spawned by spearmint write to this file as well as spearmint itself.  This enables spearmint to spawn jobs on multiple nodes in a cluster.  Of course, the writes to this file must be atomic.  When you kill spearmint, the jobs it has spawned will keep running (this is the intended behavior - the assumption is that jobs are expensive, so something that kills spearmint hopefully wouldn't stop the running jobs).  There is a caveat that spearmint may die when it is holding the lock to this `expt_grid.pkl`.  If you find that you restart spearmint and it is perpetually "waiting to lock grid..." you can safely remove the lock by typing `rm <expt_dir>/expt_grid.pkl.lock`.

**Question**: Spearmint locks up after performing the first sampling computation.  
**Answer**: Python's multiprocessing causes a deadlock on some platforms.  You can pass in a command line argument `--method-args=use_multiprocessing=0` to turn off multiprocessing and do computations serially (this is a bit slower but works just as well).
