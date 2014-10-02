'''Uses KL_divergence and kuiper test to monitor how simmilar to a uniform distriburtion is for removing different wavelengths from a spectra'''
import numpy as np
import pylab as lab
import database_utils as util
import emcee_lik
import emcee
import sys, os
import cPickle as pik
from scipy.stats import entropy as kl_diverg
from kuiper import kuiper_two as kuiper
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

def run_fit(posterior, pool, temp_save_path='unfinnish.pik'):
    '''Does fitting with emcee. Returns sampler class'''
    # make sampler and save
    nwalkers = 4 *  posterior.ndim()
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
    while True:
        for pos, prob, rstate in sampler.sample(pos, iterations=100, rstate0=rstate):
            show = 'Real run: Postieror=%e acceptance=%2.1f ESS=%2.1f'%(np.mean(prob), 100*accept, ess)
            print show
        
        pik.dump((posterior.data, sampler),
            open(temp_save_path, 'w'), 2)
        # calcuate effective sample size. quit if ess>=1000
        ess = (sampler.flatchain.shape[0]/
               np.nanmin(sampler.get_autocorr_time()))
        if ess >= 1000:
            break
        

    return sampler


def get_marg_info(sampler):
    '''makes chains into marginalize postierior and does KL and kuiper test against uniform'''
    chains = sampler.flatchain[:,:3]
    out_post = []
    out_div = []
    # sfh
    for i in range(3):
        x, y = np.histogram(chains[:,i], bins=int(np.sqrt(len(chains))))
        post = np.vstack((y[:-1], x/np.sum(x,dtype=float))).T
        out_post.append(post)
        unif = np.ones_like(post[:,1])/post[:,0].ptp()
        out_div.append([kl_diverg(post[:,1], unif), kuiper(post[:,1], unif)[0]])
    #return sfh_post, age_post, met_post, sfh_div, age_div, met_div
    return out_post[0], out_post[1], out_post[2], out_div[0], out_div[1], out_div[1]



def run_vanilla_MC(db_path, min_wave=3500, max_wave=8000):
    '''Fits only ssps. and returns KL-divergence and
    Kuiper stats from each wavelength region removed
    if bins is str will assume to make bins same number as range'''
    comm = mpi.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if rank == 0:
        param, spec, wave = get_data(db_path)
        # get min and max wavelength
        index = np.where(np.logical_and(wave >= min_wave, wave <= max_wave))[0]
        wave = wave[index]
        spec = spec[:,index]
       # check if recovery file is around
        # get data to fit
        index = np.logical_and(param[:,1]>=9, param[:,0] == 0)
        spec = spec[index]
        spec_shape = spec.shape[0]
        data_param = param[index]
        results = {}
    else:
        data = None
        wave = None
        spec_shape = None
    
    # send data and initalize pools
    pool = emcee_lik.MPIPool_stay_alive(loadbalance=True)
    wave, spec_shape = comm.bcast((wave, spec_shape))
    # start fitting
    for spec_index in xrange(spec_shape):
        if rank == 0:
            data = np.vstack((wave, spec[spec_index,:])).T
        data = comm.bcast(data)
        posterior = emcee_lik.LRG_emcee({spec_index:data}, db_path, have_dust=False,
                                        have_losvd=False)
        posterior.init()
        # workers
        if not pool.is_master():
            pool.wait(posterior)
            continue
        # make samplers and run
        sampler = run_fit(posterior, pool)
        pool.close()
        # get margionalized posteriors and kuiper,KL divergence for each
        results[spec_index] = [sampler, data, data_param[spec_index]]

    # return
    if pool.is_master():
        return results
    else:
        print 'worker is done'
        sys.exit(0)

def plot_div(path):
    '''Plots KL-divergence vs wavelength to find which parts contribue the most
    to fitting'''
    import pylab as lab
    bins, data, data_param, cur_wave, results = pik.load(open(path))
    sfh, age, metal,bins = [], [], [], []
    for wave in results:
        if wave <0:
            continue
        bins.append(wave)
        sfh.append(results[wave][-3][0])
        age.append(results[wave][-2][0])
        metal.append(results[wave][-1][0])
    bins = np.asarray(bins)
    index = bins.argsort()
    sfh = np.asarray(sfh)[index]
    age = np.asarray(age)[index]
    metal = np.asarray(metal)[index]
    bins = bins[index]
    
    lab.plot(bins, sfh, bins, age, bins, metal)
    lab.legend(('SFH' , 'Age' , 'Metals'))
    lab.xlabel('Wavelength')
    lab.ylabel('KL-divergence')
    lab.title('sfh=%f, age=%f, metal=%f'%(data_param[0],data_param[1],data_param[2]))
    lab.show()
    return bins, sfh, age, metal
    
if __name__ == '__main__':
    db_path = '/home/thuso/Phd/experements/hierarical/LRG_Stack/burst_dtau_10.db'
    results = run_vanilla_MC(db_path)
    pik.dump(a, open('test1.pik','w'),2)
