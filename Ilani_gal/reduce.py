#! /usr/bin/env python2

import pylab as lab
import numpy as nu
import cPickle as pik
import os

def load_dir(in_dir,option='age'):
    out = {}
    files = os.listdir(in_dir)
    if not in_dir.endswith('/'):
        in_dir += '/'
    line = ['-','--','-.','-s','-*','-*'] #,':','.','o','v','^','s','p','*','+','x','D','|','_']    
    color = ['b','g','r','c','m','k']
    k = []
    #decide color and marker scheme
    for i in line:
       for j in color:
           k.append(i+j)

    for i in files:
        if not i.endswith('.pik'):
            continue
        temp = pik.load(open(in_dir + i))
        param,chi = temp[-3],temp[-2]
        #find places where things have happend
        for j in param.keys():
            if len(param[j]) > 0:
                if j not in out.keys():
                    out[j] = []
                out[j].append(param[j].copy())

    for i in out.keys():
        #plot
        fig = lab.figure()
        plt = fig.add_subplot(111)
        plt.set_title('%s SSPs'%i)
        plt.set_ylabel(option,size='xx-large')
        plt.set_xlabel('Iteration',size='xx-large')
        for j in range(len(out[i])):
            #decide which index to plot
            if option.lower() == 'age':
                l = range(1,3*int(i),3)
            elif option.lower() == 'norm':
                l = range(2,3*int(i),3)
            elif option.lower() == 'metal':
                l = range(0,3*int(i),3)
            plt.plot(out[i][j][:,l], k[j])

    lab.show()

if __name__ == '__main__':
    
    import sys
    if len(sys.argv) < 2:
        print 'To run please type: ./reduce.py in_dir [parameter]'
        raise TypeError('Expected at lest 1 arguments, got 0')

    if len(sys.argv) == 2:
        in_dir = sys.argv[1]
        option = 'Age'
    else:
        in_dir = sys.argv[1]
        option = sys.argv[2]

    load_dir(in_dir,option)
