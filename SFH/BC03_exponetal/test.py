import os,sys
import cPickle as pik
#import pyfits as fits
import numpy as nu
import spectra_func as sp
import Age_date as ag
import pylab as lab
indir = '/home/thuso/Phd/experements/SFH/BC03_exponetal/models/'   
files = os.listdir(indir)
local_files = nu.array(os.listdir(os.popen('pwd').readline()[:-1]))
spect = sp.load_spec_lib('ssp_lib/')
ag.info = spect[1]
i=files[0]
spect = sp.load_spec_lib('ssp_lib/')
ag.info = spect[1]
data = nu.loadtxt(indir + i)
 #change wavelength range where galxev did dispersion
index = nu.searchsorted(data[:,0],[3330,9290])
data = data [index[0]:index[1],:]
ag.spect = nu.copy(spect[0])
fun = ag.MC_func(data, itter= 10**6)
fun.autosetup()
ag.spect = fun.spect
#Top = hy.Topologies('max')

#lab.plot(fun.data[:,0],fun.func(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))[:,1])
#print 'between'
#fun.func(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))
#print 'between'
#fun.func(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))
#print 'between'
#fun.func(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))
#print 'between'
#N,chi=fun.func_N_norm(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))
d=fun.func(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))
#print 'between'
#lab.plot(fun.data[:,0],fun.func(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))[:,1])
print fun.func_N_norm(nu.array([nu.log10(.02),9.977724,2]),nu.zeros(2),nu.array([nu.log10(250),0,0,0]))
print sum((d[:,1] - fun.data[:,1])**2)
#lab.plot(fun.data[:,0],fun.data[:,1])  
#lab.show()
'''
import numpy as nu
import pylab as lab
lab.ion()
file = open('out')
a1,a2,a3,a4 = [],[],[],[]
t=0
for i in file:
    if i.split()[0] == 'between':
        t+=1
        continue
    if t == 0:
        a1.append(nu.float64(i.split()))
    elif t ==1:
        a2.append(nu.float64(i.split()))
    elif t ==2:
        a3.append(nu.float64(i.split()))
    elif t ==3:
        a4.append(nu.float64(i.split()))
        
sum = 0
for i in xrange(len(old)):
    sum += nu.all(old[i] == new[i])
print len(old), sum
lab.plot(new);lab.plot(old)
'''
