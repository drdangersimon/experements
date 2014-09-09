
#makes data work for fitting code

import Age_date as ag
nu = ag.nu
import pyfits as fits
import cPickle as pik
import os


def make_usable(fl):
    temp = fits.open(fl)
    data = nu.zeros([temp[0].header['NAXIS1'],3])
    data[:,2] = nu.sqrt(temp[5].data)
    
    data[:,1] = temp[3].data
    data[:,0] = 10**(temp[0].header['CRVAL1'] + 
                     temp[0].header['CD1_1'] *
                     nu.arange(temp[0].header['NAXIS1']))
    #redshift
    try:
        data[:,0] = data[:,0]/(1 + float(temp[0].header['Z_FIN']))
    except KeyError:
        data[:,0] = data[:,0]/(1 + float(temp[0].header['REDSHIFT']))
    return data

if __name__ == '__main__':
    files = os.listdir('ReSpectra/')
    local_files = nu.array(os.listdir(os.popen('pwd').readline()[:-1]))
    spect = nu.copy(ag.spect)
    #fig = lab.figure()
    for i in files:
        if not i.endswith('.fits'):
            continue
        data = make_usable('ReSpectra/' + i)
        ag.spect = nu.copy(spect)
        fun = ag.MC_func(data, itter= 10**6)
        fun.autosetup()
        ag.spect = fun.spect
        param,chi,bayes = fun.run()
        pik.dump((param,chi,data),open(i[:i.find('.fits')] + '.pik','w'),2)
    
