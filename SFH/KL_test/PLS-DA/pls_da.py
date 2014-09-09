'''Uses pls-da to find how wavelengths affect different parameters'''

import numpy as np
from sklearn.cross_decomposition import PLSRegression,  CCA
import pylab as lab
import database_utils as util
import emcee_lik
import emcee
import sys
import cPickle as pik
mpi = emcee_lik.MPI

def get_data(db_path):
    '''retrives data from db.
    returns matrix of params (ssp, n param), spec (ssp, wavelength)'''

    db = util.numpy_sql(db_path)
    # get table name
    table_name = db.execute('select * from sqlite_master').fetchall()[0][1]
    # fetch all
    spec, param = [] ,[]
    for imf, model, tau, age, metal, buf_spec in db.execute('SELECT * From %s'%table_name):
        spec.append(util.convert_array(buf_spec)[:,1])
        param.append([tau, age, metal])
    
    param = np.array(param)
    spec = np.array(spec)
    return param, spec, util.convert_array(buf_spec)[:,0]

def get_correlations(param, spec, wave):
    '''Returns correlations between spec and params by wavelengths'''
    # using PLS
    pls = PLSRegression(10)
    pls.fit(spec, param)
    
    #get corretalions
    nparam = param.shape[1]
    cor = pls.coefs*np.asarray([pls.x_std_]*nparam).T
    cor /= np.tile(pls.y_std_, (cor.shape[0],1))

    return cor


def run_fit(posterior, pool, temp_save_path='unfinnish.pik'):
    '''Does fitting with emcee. Returns sampler class'''
    # make sampler and save
    nwalkers = 2 *  posterior.ndim()
    sampler = emcee.EnsembleSampler(nwalkers, posterior.ndim(), 
                                    posterior, pool=pool)
    pik.dump((posterior.data, sampler),
             open(temp_save_path,'w'),2)
    # burn-in
    pos0 = posterior.inital_pos(nwalkers)
    autocorr, accept = 0, 0
    for pos, prob, rstate in sampler.sample(pos0, iterations=500):
        show = 'Burn-in: Postieror=%e acceptance=%2.1f autocorr=%2.1f'%(np.mean(prob), 100*accept,autocorr)
        print show
    # get ready for real run
    sampler.reset()
    ess = 0.
    while True
        for pos, prob, rstate in sampler.sample(pos, iterations=100, rstate0=rstate):
        show = 'Real run: Postieror=%e acceptance=%2.1f ESS=%2.1f'%(np.mean(prob), 100*accept, ess)
        print show
        
        pik.dump((posterior.data, sampler),
            open(temp_save_path, 'w'), 2)
        # calcuate effective sample size. quit if ess>=1000
        ess = (sampler.flatchain.shape[0]/
               np.nanmin(hange_sampler.get_autocorr_time()))
        if ess >= 1000:
            break
        

    return sampler



def fit_age(db_path, param, min_wave=3500, max_wave=10000):
    '''Fits a spectra 2 times, once with all the wavelngths, and another
    without 0 correlation parts. returns chains'''
    assert param.lower() in ['age', 'sfh', 'metals', 'z'], 'input param is wrong'
    comm = mpi.COMM_WORLD
    rank = comm.rank
    size = comm.size
    #assert size > 1, 'Must run with mpi'
    if rank == 0:
        # find informative wavelengths
        params, spec, wave = get_data(db_path)
        wave_index = np.where(np.logical_and(wave >= min_wave,
                                              wave <= max_wave))[0]
        spec = spec[:, wave_index]
        wave = wave[wave_index]
        spec_info = get_correlations(params, spec, wave)
        # get 100 data points with highest correlation for parameter
        if param.lower() == 'age':
            param = 1
        elif param.lower() == 'sfh':
            param = 0
        elif param.lower() in ['metals', 'metal', 'z']:
            param = 2
        else:
            raise ValueError('param must me age, metals or sfh.')
        info_wave = np.argsort(spec_info[:,param]**2/
                                np.sum(spec_info[:,param]**2))[-100:]
        # choose random spectra
        spec_index = np.random.randint(spec.shape[0])
        data_param = params[spec_index]
        data = np.vstack((wave, spec[spec_index,:])).T
    else:
        data = None
    # get data from root
    data = comm.bcast(data, root=0)
    # make liklihood
    posterior = emcee_lik.LRG_emcee({'no_cor':data}, db_path, have_dust=False,
                                    have_losvd=False)
    posterior.init()
    pool = emcee_lik.MPIPool_stay_alive(loadbalance=True)
    if not pool.is_master():
        # fits 2 times so calls twice and exit
        pool.wait(posterior)
        print 'worker'
        data = comm.bcast(data, root=0)
        posterior = emcee_lik.LRG_emcee({'cor':data}, db_path, have_dust=False,
                                        have_losvd=False)
        posterior.init()
        pool.wait(posterior)
        sys.exit(0)
    no_change_sampler = run_fit(posterior, pool)
    pool.close()
    
    # changed sampler
    data = data[info_wave,:]
    data = comm.bcast(data, root=0)
    posterior = emcee_lik.LRG_emcee({'cor':data}, db_path, have_dust=False,
                                        have_losvd=False)
    posterior.init()
    change_sampler = run_fit(posterior, pool)
    pool.close()
    return data_param, no_change_sampler, change_sampler
 
    
if __name__ == '__main__':
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    # age
    params, no_change_sampler, change_sampler = fit_age(db_path, 'age')
    # save
    pik.dump((params, no_change_sampler, change_sampler), open('try.pik','w'),
             2)
    #done
    print 'done'
