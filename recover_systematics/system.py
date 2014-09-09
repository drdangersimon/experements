


import numpy as nu
import pylab as lab
import pca 
import os
import cPickle as pik

#handles chains fitted for recovering of systematics
def load_best_fit_param(indir):
    '''returns best fit param values from dir. does in form of a dict'''
    if not indir.endswith('/'):
        indir += '/'
    files = os.listdir(indir)
    out = {}
    for i in files:
        #if not a .pik file skip
        if not i.endswith('.pik'):
            continue
        temp = pik.load(open(indir+i))
        key = guess_best_fit(temp[0],temp[1])

        out[i[:i.find('.pik')]] = temp[0][key][(temp[1][key].min() == 
                                                temp[1][key])][0]

    return out

def guess_best_fit(param,chi):
    '''does "baysiean" model selsction and returns best fit key'''
    best_fit = ['0', 1., nu.inf]
    for i in param.keys():
        if len(param[i]) == 0:
            continue
        if best_fit[2] >= chi[i].min(): #chi must be less
            if len(param[i])/best_fit[1] > 1/3.:
                if int(i) > int(best_fit[0]): #if has more params
                    #penalize it and see if still better
                    if (chi[i].min() * 
                        2.**((3*(int(i) -int(best_fit[0])))) < 
                        best_fit[2]): 
                        best_fit = [i,len(param[i]),chi[i].min()]
                else: 
                    best_fit = [i,len(param[i]),chi[i].min()]
    print 'Using best fit param with %s bins and min chi %f' %(best_fit[0], 
                                                               best_fit[2])

    return best_fit[0]
 
def residuals(param,indir):
    '''outputs residuals from best fit values.
    param should be a dict with best fit params in it, indir should
    show where data spectrum is'''
    if not indir.endswith('/'):
        indir += '/'
    files = os.listdir(indir)
    print 'stellar lib is %s' %files[0]
    import Age_date as ag
    out = {}
    spect = nu.copy(ag.spect)
    for i in files:
        #if not a .pik file skip
        if not i.endswith('.pik') or not param.has_key(i[:i.find('.pik')]):
            continue
        temp = pik.load(open(indir+i))
        fun = ag.MC_func(temp[2])
        ag.spect = nu.copy(spect)
        bins = (len(param[i[:i.find('.pik')]]) - 2) / 3
        temp_data = ag.plot_model(param[i[:i.find('.pik')]],temp[2],bins,False)
        #incase shapes aren't the same
        if fun.data.shape[0] - temp_data.shape[0] == 0:
            out[i[:i.find('.pik')]] = nu.vstack((fun.data[:,0],
                                                 (fun.data[:,1] - 
                                                  ag.normalize(fun.data,
                                                               temp_data[:,1])
                                                  * temp_data[:,1]))).T
        elif fun.data.shape[0] - temp_data.shape[0] < 0:
            k =temp_data.shape[0] - fun.data.shape[0]
            out[i[:i.find('.pik')]] = nu.vstack((fun.data[:,0],
                                                 (fun.data[:,1] - 
                                                  ag.normalize(fun.data,
                                                               temp_data[k:,1])
                                                  * temp_data[k:,1]))).T
        elif fun.data.shape[0] - temp_data.shape[0] > 0: 
            k = fun.data.shape[0] - temp_data.shape[0]
            out[i[:i.find('.pik')]] = nu.vstack((fun.data[k:,0],
                                                 (fun.data[k:,1] - 
                                                  ag.normalize(fun.data,
                                                               temp_data[:,1])
                                                  * temp_data[:,1]))).T

    return out

def make_ready_pca(resid):

    '''puts in pca format'''
    wave_bins = []
    for i in resid.values():
        wave_bins.append(i[:,0])
    wave_bins = nu.unique(nu.concatenate(wave_bins))
    out = nu.zeros([len(resid.keys()),len(wave_bins)])
    #put into out array
    vals = resid.values()
    for i in xrange(len(vals)):
        for j in vals[i]:
            out[i,nu.searchsorted(wave_bins,j[0])] = j[1]
        
    return out
