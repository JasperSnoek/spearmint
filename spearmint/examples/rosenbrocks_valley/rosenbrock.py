import math

def rosenbrocks_valley(xs):
    sum = 0
    last_x = xs[0]

    for i in xrange(1, len(xs)):
        sum += (100 * math.pow((xs[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)

    return sum

def main(job_id, params):
  xs = params['X']
  res = rosenbrocks_valley(xs)

  print "Rosenbrock's Valley in %d dimensions" % (len(xs))
  print "\tf(",
  print xs,
  print ") = %f" % (res)

  return rosenbrocks_valley(xs)


if __name__ == "__main__":
    main(3, {'X': [1.73, 0.2]})
