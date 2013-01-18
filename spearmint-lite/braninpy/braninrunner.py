import branin
"""
Jasper Snoek
This is a simple script to help demonstrate the functionality of
spearmint-lite.  It will read in results.dat and fill in 'pending'
experiments.
"""
if __name__ == '__main__':
    resfile = open('results.dat','r')
    newlines = []
    for line in resfile.readlines():
        values = line.split()
        if len(values) < 3:
            continue
        val = values.pop(0)
        dur = values.pop(0)
        X = [float(values[0]), float(values[1])]
        print X
        if (val == 'P'):
            val = branin.branin(X)
            newlines.append(str(val) + " 0 " 
                            + str(float(values[0])) + " " 
                            + str(float(values[1])) + "\n")
        else:
            newlines.append(line)

    resfile.close()
    outfile = open('results.dat','w')
    for line in newlines:
        outfile.write(line)
