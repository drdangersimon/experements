import Age_date as ag
import Age_hybrid as hy
import pylab as lab
import os
import cPickle as pik
nu=lab.np
'''does tests against rjmcmc and 2 types of swarm functions'''

comm = hy.mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
'''
#test over number of ssp's
for i in range(1,6):
    if rank == 0:
        #make data and send it
        data,info,weight,dust = ag.iterp_spec(i,lam_min=4000, lam_max=8000)
        data_len = nu.array(data.shape)
        comm.bcast(data_len,0)
    else:
        data_len = nu.zeros(2,dtype=int)
        comm.bcast(data_len,0)
        data = nu.zeros(data_len)
    data = comm.bcast(data, 0)
    Top = hy.Topologies(max)
    fun = ag.MC_func(data)
    fun.autosetup()
    host = os.popen('hostname').read()
    #else:
    print 'starting hybrid on %s'%host
    param,chi,bayes = hy.root_run(fun.send_class, Top, itter=3*10**6, k_max=16, func=hy.vanilla)
    if rank ==0:
            #save
        pik.dump((param,chi,data,info,weight,dust), open('vanila_swarm_%d_ssp.pik'%i,'w'),2)
    print 'starting vanilla on %s'%host
    param,chi,bayes = hy.root_run(fun.send_class, Top, itter=3*10**6, k_max=16, func=hy.hybrid)
    if rank ==0:
            #save
        pik.dump((param,chi,data,info,weight,dust), open('hybrid_swarm_%d_ssp.pik'%i,'w'),2)
'''
#monte carlo run
data,info,weight,dust = ag.iterp_spec(1,lam_min=4000, lam_max=8000)
fun=ag.MC_func(data)
fun.autosetup()
age_unq=fun._age_unq
metal_unq=fun._metal_unq
dust_range=nu.array([0,4])
sigma=nu.array([0,3])
bins=2
#redshift=nu.array([0,1])
#h3=nu.array([-6,6])
#h4=nu.array([-6,6])
for i in range(100):
    if rank == 0:
        param = nu.zeros(3*bins)
        param[range(0,bins*3,3)] = nu.random.rand(bins) * metal_unq.ptp() + metal_unq.min()
        param[range(1,bins*3,3)] = nu.random.rand(bins) * age_unq.ptp() + age_unq.min()
        param[range(2,bins*3,3)] = nu.random.rand(bins)*100
        #put in bad parts of param
        param[range(1,bins*3,3)] = [5.9,10.2]
        param[range(2,bins*3,3)]  = [100., 25.]
        dust = nu.random.rand(2)*dust_range.ptp() + dust_range.min()
        #losvd[dispersion,redshift,h3,h4]
        losvd = nu.zeros(4)
        losvd[0]= nu.random.rand()*sigma.ptp() + sigma.min()
        #losvd[1] = nu.random.rand()
        #losvd[0]
        #losvd[0]
        print param,dust,losvd
        data = fun.func(param,dust,losvd)
        data_len = nu.array(data.shape)
        comm.bcast(data_len,0)
    else:
        data_len = nu.zeros(2,dtype=int)
        comm.bcast(data_len,0)
        data = nu.zeros(data_len)
    data = comm.bcast(data, 0)
    Top = hy.Topologies(max)
    fun = ag.MC_func(data)
    fun.autosetup()
    host = os.popen('hostname').read()
    print 'starting vanilla on %s'%host
    Param,chi,accept = hy.root_run(fun.send_class, Top, itter=10**6, k_max=16, func=hy.vanilla)
    if rank ==0:
            #save
        pik.dump((Param,chi,data,accept,param,dust,losvd), open('vanila_monte_carlo_%d_ssp.pik'%i,'w'),2)
    print 'starting hybrid on %s'%host
    Top = hy.Topologies(max)
    Param,chi,accept= hy.root_run(fun.send_class, Top, itter=10**6, k_max=16, func=hy.hybrid)
    if rank ==0:
            #save
        pik.dump((Param,chi,data,accept,param,dust,losvd), open('hybrid_monte_carlo_%d_ssp.pik'%i,'w'),2)

'''
data,info,weight,dust = ag.iterp_spec(1,lam_min=4000, lam_max=8000)
fun=ag.MC_func(data)
fun.autosetup()
age_unq=fun._age_unq
metal_unq=fun._metal_unq
dust_range=nu.array([0,4])
sigma=nu.array([0,3])
param = nu.zeros(3)
param[0] = nu.random.rand() * metal_unq.ptp() + metal_unq.min()
param[1] = nu.random.rand() * age_unq.ptp() + age_unq.min()
param[2] = nu.random.rand()*100
dust = nu.random.rand(2)*dust_range.ptp() + dust_range.min()
losvd = nu.zeros(4)
losvd[0]= nu.random.rand()*sigma.ptp() + sigma.min()
lab.plot(fun.func(param,dust,losvd)[:,1])
data = fun.func(param,dust,losvd)
'''
