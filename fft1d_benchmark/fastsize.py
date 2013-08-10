#! /usr/bin/python

import sys

#========================================================================
def filter_slow_size(f, i):
    final = []
    lines = []
    cdata = []
    ldata = [0, 0, 0]
    for line in f:
        if line[0]=='#':
            continue;
        lines.append(line)
    lines.reverse()
    for line in lines:
        cdata = line.split()
        if(float(cdata[i])>float(ldata[i])):
            ldata=cdata
            final.append(line.strip('\n'))
    final.reverse()
    for line in final:
        cdata=line.split()
        print "{:>8}  {:>12}".format(cdata[0], cdata[i])

#========================================================================
if __name__ == "__main__":
    if(len(sys.argv)<2):
        print "The program filter out slow FFT size, take two arguments!"
        print "Usage: ",sys.argv[0]," x87.log index"
        print "  index=1 for r2c transform, index=2 for c2c transform"
    else :
        with open(sys.argv[1], 'r') as f:
            filter_slow_size(f, int(sys.argv[2]))
