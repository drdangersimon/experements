
#runs fitting rjmcmc from models using spectra library in ssp_lib

import Age_date as ag
import Age_hybrid as hy
import os,sys
import cPickle as pik
import pyfits as fits
import numpy as nu
import spectra_func as sp


if __name__ == '__main__':
    indir = 'models/'
    outdir = sys.argv[1]
    if not outdir[-1] == '/':
        outdir += '/'
    files = os.listdir(indir)
    local_files = nu.array(os.listdir(os.popen('pwd').readline()[:-1]))
    spect = sp.load_spec_lib('ssp_lib/')
    ag.info = spect[1]
    #fig = lab.figure()
    for i in files:
        if not i.endswith('.spec'):
            continue
        data = nu.loadtxt(indir + i)
        print i
        #change wavelength range where galxev did dispersion
        index = nu.searchsorted(data[:,0],[3330,9290])
        data = data [index[0]:index[1],:]
        ag.spect = nu.copy(spect[0])
        fun = ag.MC_func(data, itter= 10**6)
        fun.autosetup()
        ag.spect = fun.spect
        Top = hy.Topologies('max')
        param,chi,bayes = hy.root_run(fun.send_class, Top, itter=10**6, k_max=16, func=hy.vanilla)
        #param,chi,bayes = fun.run()
        pik.dump((param,chi,data),open(outdir+i[:i.find('.spec')] + '.pik','w'),2)
   
