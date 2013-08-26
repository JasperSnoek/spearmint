def dejong(x,y):
    return x*x + y*y

# Write a function like this called 'main'
def main(job_id, params):
  x = params['X'][0]
  y = params['Y'][0]
  res = dejong(x, y)
  print "De Jong's function in 2D:"
  print "\tf(%.2f, %0.2f) = %f" % (x, y, res)
  return dejong(x, y)


if __name__ == "__main__":
    main(23, {'X': [1.2], 'Y': [4.3]})
