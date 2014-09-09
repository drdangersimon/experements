#!/usr/bin/env python

#checks uncertatny of differnt ssp at different ages


import scipy.stats as sci
import Age_MCMC as mc
nu=mc.nu

lib_vals=mc.get_fitting_info(mc.lib_path)
lib_vals[0][:,0]=10**nu.log10(lib_vals[0][:,0]) #to keep roundoff error constistant
metal_unq=nu.log10(nu.unique(lib_vals[0][:,0]))
age_unq=nu.unique(lib_vals[0][:,1])


def create_hists(num_ssp=1):
    #tells minimum uncertanty for different ages and trys and groups them when they recover the same age and normalization
    assert num_ssp==1, 'not ready for more than 1 ssp'

    age=[age_unq.min()+.01]
    metal=[metal_unq.mean()]
    norm=[1000.]
    data,info,weight=mc.own_array_spect(age, metal, norm)
    param,chi=mc.MCMC_multi(data,10**5,num_ssp)
    recoverd=[];recoverd.append([nu.mean(param[:,1]),nu.std(param[:,1])]) 
    age.append(age[-1]+nu.std(param[:,1]))
#    if chi.min()>1.:
#        print 'did not converge trying again'

    while age[-1]<age_unq.max():
        print 'recoveing at %f Gyrs' %age[-1]
        data,info,weight=mc.own_array_spect([age[-1]], metal, norm)
        param,chi=mc.MCMC_multi(data,10**5,num_ssp)
        recoverd.append([nu.mean(param[:,1]),nu.std(param[:,1])])
        if nu.std(param[:,1])<10**-2:
            age.append(age[-1]+10**-1)
        else:
            age.append(age[-1]+nu.std(param[:,1]))

    return nu.array(age[1:]),nu.array(recoverd)

def create_hists_werror(stnr,num_ssp=1):
    #tells minimum uncertanty for different ages and trys and groups them when they recover the same age and normalization
    #stnr is signal to noise ratio give a constant uncertaty
    assert num_ssp==1, 'not ready for more than 1 ssp'

    age=[age_unq.min()+.01]
    metal=[metal_unq.mean()]
    norm=[1000.]
    data,info,weight=mc.own_array_spect(age, metal, norm)
    if stnr>sci.signaltonoise(data[:,1]):
        print 'Warrning: max STNR is %f and will use that' %sci.signaltonoise(data[:,1])
        stnr=sci.signaltonoise(data[:,1])
    data=nu.hstack((data,data/stnr))[:,[0,1,3]]
    param,chi=mc.MCMC_multi(data,10**5,num_ssp)
    recoverd=[];recoverd.append([nu.mean(param[:,1]),nu.std(param[:,1])]) 
    age.append(age[-1]+nu.std(param[:,1]))
#    if chi.min()>1.:
#        print 'did not converge trying again'

    while age[-1]<age_unq.max():
        print 'recoveing at %f Gyrs' %age[-1]
        data,info,weight=mc.own_array_spect([age[-1]], metal, norm)
        param,chi=mc.MCMC_multi(data,10**5,num_ssp)
        recoverd.append([nu.mean(param[:,1]),nu.std(param[:,1])])
        if nu.std(param[:,1])<10**-2:
            age.append(age[-1]+10**-1)
        else:
            age.append(age[-1]+nu.std(param[:,1]))

    return nu.array(age[1:]),nu.array(recoverd)

