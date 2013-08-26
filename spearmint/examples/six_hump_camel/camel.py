import math

def camel(x,y):
    x2 = math.pow(x,2)
    x4 = math.pow(x,4)
    y2 = math.pow(y,2)

    return (4.0 - 2.1 * x2 + (x4 / 3.0)) * x2 + x*y + (-4.0 + 4.0 * y2) * y2


def main(job_id, params):
  x = params['X'][0]
  y = params['Y'][0]
  res = camel(x, y)
  print "The Six hump camel back function:"
  print "\tf(%.4f, %0.4f) = %f" % (x, y, res)
  return camel(x, y)


if __name__ == "__main__":
    main(23, {'X': [0.0898], 'Y': [-0.7126]})
