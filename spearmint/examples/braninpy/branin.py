import numpy as np
import sys
import math
import time

def branin(x):
  x
  if x[0] <0 or x[0] > 1:
    return np.NaN

  if x[1] <0 or x[1] > 1:
    return np.NaN

  x[0] = x[0]*15
  x[1] = (x[1]*15)-5

  y = np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0] - 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10;

  result = y

  print result
  return result

# Write a function like this called 'main'
def main(job_id, params):
  print 'Anything printed here will end up in the output directory for job #:', str(job_id)
  print params
  return branin(params['X'])
