import utils
import likelihood_class as lik
from glob import glob
import numpy as nu
import os
import thuso_quick_fits as T
import pylab as lab
'''Spectral Stacker.

Take HI profiles and tries different bayesan methods of stacking'''


def quick_fits(path='subset'):
    '''takes all spectra and co-adds them and returns total, average
    mass of all galaxies'''
    path = os.path.join(path,'')
    files = glob(path + '*.txt')
    #param,name
    out_param = []
    func = lambda x,p: utils.busy(x,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7])
    const = nu.array([[   0.,   nu.inf],
       [ -nu.inf,   nu.inf],
       [ -nu.inf,   nu.inf],
       [   0.,   nu.inf],
       [   0.,  500.],
       [   0.,  500.],
       [   0.,   nu.inf],
       [   1.,   10.]])
    param = nu.array([40.0, 1, 1, 5, 111, 120, 5, 2])
    for spectra in files:
        #get data
        data = load_spec(spectra)
        #fit
        ### need to look at the noise
        _,_,b,c = T.quick_cov_MCMC(data[:,0],data[:,1],param,func,const)
        #save
        out_param.append([b,spectra])
        #plot
        lab.figure()
        lab.plot(data[:,0],data[:,1],data[:,0],func(data[:,0],b),label=spectra)
        lab.show()
    return out_param


def load_spec(path):
    '''Loads spectra'''
    data = nu.loadtxt(path, comments=';')
    return data
    

#good one spectra = 'subset/610622.txt'
