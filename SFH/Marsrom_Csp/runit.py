#!/usr/bin/env /share/apps/bin/python2.7
#does fit of all globular cluster fits from dir "LIST"

import Age_date as ag
import os
import cPickle as pik
#import pyfits as fits
import numpy as nu


def make_usable(fl):
    temp = nu.loadtxt(fl)
    ssp = {}
    age = nu.unique(temp[:,0])
    for i in age:
        ssp[str(i)]= temp[temp[:,0] == i,1:]
    return ssp

if __name__ == '__main__':
    Dir = 'sed/'
    files = os.listdir(Dir)
    local_files = nu.array(os.listdir(os.popen('pwd').readline()[:-1]))
    spect = nu.copy(ag.spect)
    #fig = lab.figure()
    for i in files:
        if not i.endswith('.sed_agb'):
            continue
        ssp = make_usable(Dir + i)
        for j in ssp.keys():
            data = ssp[j]
            ag.spect = nu.copy(spect)
            fun = ag.MC_func(data, itter= 10**6)
            fun.autosetup()
            ag.spect = fun.spect
            param,chi,bayes = fun.run()
            pik.dump((param,chi,data),open(i[:i.find('.sed_agb')] + '_age_%s.pik'%j,'w'),2)
   
