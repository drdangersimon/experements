#!/usr/bin/env python

#spectra stacking test program
import numpy as nu
import Age_date as ag


def create_stn_spect(spect,stn):
    #adds gaussian noise to spectra for a average signal to noise
    signal=nu.mean(spect[:,1])
    noise=signal/stn
    out=nu.copy(spect)
    out[:,1]+=nu.random.randn(out[:,1].shape[0])*noise
    return out
def vanilla_stack(list_spect):
    #assume all wavelengths are same, just takes average of spects
    signal=nu.zeros_like(list_spect[0])
    signal[:,0]=nu.copy(list_spect[0][:,0])
    n=float(len(list_spect))
    for i in list_spect:
        signal[:,1]+=i[:,1]/n
        
    return signal
