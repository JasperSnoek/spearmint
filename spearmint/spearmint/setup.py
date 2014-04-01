import glob
import os
from distutils.core import setup

# TODO: probably best to put all python files except for the top level run
# script into a package (sub-directory), but for now we can do this.
MODULES = ['ExperimentGrid', 'gp', 'helpers', 'Locker', 'runner', 'sobol_lib', 'spearmint_pb2', 'util']

setup(name='spearmint',
      description="Practical Bayesian Optimization of Machine Learning Algorithms",
      author="Jasper Snoek, Hugo Larochelle, Ryan P. Adams",
      url="https://github.com/JasperSnoek/spearmint",
      version='1.0',
      license='GPLv3',
      packages=['driver', 'chooser'],
      py_modules=MODULES
     )
