#!/usr/bin/env /share/apps/bin/python2.7
#does fit of all globular cluster fits from dir "LIST"

import Age_date as ag
import os
import cPickle as pik
import pyfits as fits
import numpy as nu
import Age_hybrid as hy

def make_usable(fl):
    temp = fits.open(fl)
    data = nu.zeros([temp[0].header['NAXIS1'],2])
    #data[:,2] = nu.sqrt(temp[5].data)
    
    data[:,1] = temp[0].data
    try:
        data[:,0] = (temp[0].header['CRVAL1'] + 
                     temp[0].header['CD1_1'] *
                     nu.arange(temp[0].header['NAXIS1']))
    except: 
        data[:,0] = (temp[0].header['CRVAL1'] +
                     temp[0].header['CDELT1'] *
                     nu.arange(temp[0].header['NAXIS1']))
    #redshift
    '''try:
        data[:,0] = data[:,0]/(1 + float(temp[0].header['Z_FIN']))
    except KeyError:
        data[:,0] = data[:,0]/(1 + float(temp[0].header['REDSHIFT']))'''
    #remove zeros
    data = data[data[:,1].nonzero()[0], :]
    return data

if __name__ == '__main__':
    Dir = 'Central_spectra/'
    files = os.listdir(Dir)
    local_files = os.listdir('results/')
    spect = nu.copy(ag.spect)
    #fig = lab.figure()
    for i in files:
        if not i.endswith('.fits'):
            continue
        if i[:i.find('.fits')] + '.pik' in local_files:
            continue
        data = make_usable(Dir + i)
        ag.spect = nu.copy(spect)
        fun = ag.MC_func(data, itter= 10**6,spec_lib='M11_Miles_cha.splib')
        fun.autosetup()
        ag.spect = fun.spect
        top = hy.Topologies('ring')
        top.comm_world.barrier()
        print i
        #param,chi,bayes = fun.run()
        param,chi,bayes = hy.root_run(fun.send_class, top, itter=10**6, burnin=5000 , k_max=10, func=hy.vanilla)
        if top.comm_world.rank == 0:
            print 'saving to %s' %('results/' + i[:i.find('.fits')] + '.pik')
            pik.dump((param,chi,data),open('results/' + i[:i.find('.fits')] + '.pik','w'),2)
            os.popen('mkdir ' + i[:i.find('.fits')] + '; mv *.pik '+i[:i.find('.fits')]  + '/')
            #os.popen('rm *.pik')
            print 'Done saving to %s' %('results/' + i[:i.find('.fits')] + '.pik')
            os.popen('killall ipython')
