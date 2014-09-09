import numpy as nu
import scipy.stats as stats_dist
from scipy.cluster.hierarchy import fcluster,linkage
from mpi4py import MPI as mpi
import os, sys
#from astropy.stats.funcs import sigma_clip
from time import time, sleep
import pylab as lab
from scipy.optimize import fmin_powell
'''Test Likelihood function for MCMC pool and also creates data
'''

class MCMC_POOL(object):

    '''exmaple class for use with RJCMCM or MCMC program, all methods are
    required and inputs are required till the comma, and outputs are also
    not mutable. The body of the class can be filled in to users delight'''
    def make_gaussian(self,n_points=100,real_k=None):
        '''Creates data and stores real infomation for model
        y = sum_i^k w_i*N(mu_i,sigma_i)'''
        if not real_k:
            self._k = nu.random.randint(1,10)
        else:
            self._k = real_k
        self._max_order = 10
        self._min_order = 1
        self._order = self._k
        #generate real parametes from normal [mu,sigma] sigma>0
        self._param = nu.random.randn(self._k,2)*9 
        self._param[:,1] = nu.abs(self._param[:,1])
        self._param = self._param[self._param[:,0].argsort()]
        #generate points
        param = self._param.ravel()
        x,y = nu.random.randn(n_points)*10, nu.zeros(n_points)
        x.sort()
        for i in xrange(0,self._k*2,2):
            y += stats_dist.norm.pdf(x,param[i],nu.abs(param[i+1]))
        y *= 1000

        self.data = nu.vstack((x,y)).T
        #set prior values
        self.mu,self.var = 0.,9.
    
    def __init__(self,n_points=100,real_k=None,rank=mpi.COMM_WORLD.rank,size=mpi.COMM_WORLD.size):
        '''(Example_lik_class,#user defined) -> NoneType or userdefined

        initalize class, can do whatever you want. User to define functions'''
        #return #whatever you want or optional
        #needs to have the following as the right types
        self.make_gaussian(n_points,real_k)
        #do quick threshholding source detection for prior
        #clipped_data = sigma_clip(data,iters=None)
        #Z=linkage(a,method='centroid')
        #fcluster

        self.rank = rank
        self.size = size
        self.comm = mpi.COMM_WORLD
        self.t = []
        #find non root workers and give them a flag for an infinite loop
        if rank != 0:
            self.only_lik = True
            #make persitant comunitcators
            #recive from root
            self.recv_list = []
            self.sigma = nu.zeros((size,10,10))
            self.param = nu.zeros((size,10))
            self.send_sigma = nu.identity(10)
            self.loglik = nu.zeros((size,1)) - nu.inf
            
            self.recv_list.append(self.comm.Recv_init((self.sigma[rank,:,:],mpi.DOUBLE), 0,0))
            self.time = nu.zeros((size,1)) + time()
            self.recv_list.append(self.comm.Recv_init((self.param[rank],mpi.DOUBLE), 0,1))
            #send to root
            self.send_list = []
            self.send_list.append(self.comm.Send_init((self.loglik[rank],mpi.DOUBLE),0,2))
            self.send_list.append(self.comm.Send_init((self.param[rank],mpi.DOUBLE),0,3))
            self.send_list.append(self.comm.Send_init((self.time[rank],mpi.DOUBLE),0,4))
        else:
            #pool {time,[param,lik]}
            self.pool = {}
            self.only_lik = False
            #make persitant comunitcators
            self.sigma = nu.zeros((self.size,20,20))
            self.send_sigma = nu.identity(20)
            self.param = nu.zeros((self.size,20))
            self.loglik = nu.zeros((self.size,1))
            self.time = nu.zeros((self.size,1)) + time()
            self.recv_param = nu.zeros((self.size,20))
            self.recv_list = {}
            self.recv_list['param'] = []
            self.recv_list['lik'] = []
            self.recv_list['time'] = []
            self.send_list = []
            
            for i in range(1,size):
                #send state to workers
                self.send_list.append(self.comm.Send_init((self.sigma[i,:,:],mpi.DOUBLE), i,0))
                self.send_list.append(self.comm.Send_init((self.param[i],mpi.DOUBLE), i,1))
                #recive param and lik from workers
                self.recv_list['param'].append(self.comm.Recv_init((self.recv_param[i],mpi.DOUBLE), i , 3))
                self.recv_list['lik'].append(self.comm.Recv_init((self.loglik[i],mpi.DOUBLE),i,2))
                self.recv_list['time'].append(self.comm.Recv_init((self.time[i],mpi.DOUBLE),i,4))
            
        self.models = {}
        for i in range(1,11):
            self.models = {str(i):str(['mu,sigma']*i)} #initalizes models
        self._multi_block = False


    def proposal(self,mu,sigma):
        '''(Example_lik_class, ndarray,ndarray) -> ndarray
        Proposal distribution, draws steps for chain. Should use a symetric
        distribution'''
        
        #return up_dated_param
        #save curent sigma
        if self.rank ==0 :
            self.send_sigma = sigma
        return nu.random.multivariate_normal(mu,sigma)

    def lik_calc(self,param,bins):
        '''(Example_lik_class, ndarray) -> float
        This calculates the likelihood for a pool of workers.
        If rank == 0, send curent state to workers and wait for liks to be sent
        If rank != 0, recive current state, do trial move and start lik calc. This guy will never leave this function till the end of program
        '''
        #worker
        i = 0
        
        while self.rank != 0:
            #recive curent state
            t = time()
            mpi.Prequest.Startall(self.recv_list)
            mpi.Prequest.Waitall(self.recv_list)
            #calc likelihood
            #self.param[self.rank] = nu.random.rand(10)
            '''if i == 0:
                sleep(self.rank**2)'''
            self.time[self.rank] = time() - self.time[0]
            #make test step
            self.param[self.rank] = self.proposal(self.param[self.rank],self.send_sigma)
            
            self.loglik[self.rank] = self.data.lik_cal(self.param[self.rank])
            i+=1
            while time() -t < 5:
                pass
            #send back to root
            mpi.Prequest.Startall(self.send_list)
            mpi.Prequest.Waitany(self.send_list)
            #check if done
            #print self.rank, time() -t
            
            
        #root
        #send curent state to workers
        t= time()
        for i in range(1,self.size):
            self.param[i] = param[bins]
            self.sigma[i,:,:] = self.send_sigma
        mpi.Prequest.Startall(self.send_list)
        mpi.Prequest.Waitany(self.send_list)
        if len(self.pool.keys()) < 25:
            #recive state from workers, add new to pool
            mpi.Prequest.Startall(self.recv_list['lik'])
            mpi.Prequest.Startall(self.recv_list['param'])
            mpi.Prequest.Startall(self.recv_list['time'])
            #print time() -t,1
            mpi.Prequest.Waitany(self.recv_list['lik'])
            mpi.Prequest.Waitany(self.recv_list['param'])
            mpi.Prequest.Waitany(self.recv_list['time'])
        #print time() -t,2
        #add new ones to pool
        i = 0
        #print self.time
        #print '##############'
        while len(self.pool.keys()) == 0 or i == 0:
            i+=1
            
            for i in range(1,len(self.time)):
                #if param hasn't changed in a while don't add
                if not self.pool.has_key(str(self.time[i])):
                    self.pool[str(self.time[i])] = (nu.copy(self.recv_param[i]),
                                                    nu.copy(self.loglik[i]))
                    #print i,str(self.time[i]), len(self.pool.keys())
                
        #choose lik from pool
        
        key = nu.random.choice(self.pool.keys())
        param[bins],loglik = self.pool.pop(key)
        if loglik == 0:
            loglik = -nu.inf
        #return to main program
        #print nu.mean(self.t)*10**5/3600.
        self.t.append(time() -t)
        return loglik
        #return loglik
        

    def prior(self,param,bins):
        '''(Example_lik_class, ndarray) -> float
        Calculates log-probablity for prior'''
        #return logprior
        out = 0.
        #mean prior
        for i in param[bins][:int(bins)*2][slice(0,-1,2)]:
            out += stats_dist.norm.logpdf(i,loc=self.mu,scale=self.var)
        #var prior
        for i in param[bins][:int(bins)*2][slice(1,param[bins][:int(bins)*2].size,2)]: 
            if i < 0:
                out += -nu.inf
            else:
                out += stats_dist.norm.logpdf(i,loc=self.mu,scale=self.var)
        return out.sum()


    def model_prior(self,model):
        '''(Example_lik_class, any type) -> float
        Calculates log-probablity prior for models. Not used in MCMC and
        is optional in RJMCMC.'''
        #return log_model
        return 0.

    def initalize_param(self,model):
        '''(Example_lik_class, any type) -> ndarray, ndarray

        Used to initalize all starting points for run of RJMCMC and MCMC.
        outputs starting point and starting step size'''
        #return init_param, init_step
        #find most likeliy place
        n_param = int(model) * 2
        param = nu.zeros_like(self.param[self.rank])
        param[:n_param] = nu.random.randn(n_param) * 9
        sigma = nu.identity(len(param))
        return param, sigma
        
    def step_func(self,step_crit,param,step_size,model):
        '''(Example_lik_class, float, ndarray or list, ndarray, any type) ->
        ndarray

        Evaluates step_criteria, with help of param and model and 
        changes step size during burn-in perior. Outputs new step size
        '''
        if step_crit > .60:
            step_size[model] *= 1.05
        elif step_crit < .2 and nu.any(step_size[model].diagonal() > 10**-6):
            step_size[model] /= 1.05
        #return new_step
        if len(param) % 200 == 0 and len(param) > 0.:
            temp = nu.cov(self.list_dict_to(param[-2000:]).T)
            #make sure not stuck
            if nu.any(temp.diagonal() > 10**-6):
                step_size[model] = temp
        
        return step_size[model]

    def birth_death(self,birth_rate, bins, active_param):
        if birth_rate > nu.random.rand() and bins+1 != self._max_order :
            #birth
            attempt = True #so program knows to attempt a new model
            #create step
            temp_bins = bins + 1 #nu.random.randint(bins+1,self._max_order)
            #split move params 
            index = [nu.random.randint(0,bins)]
            mu,sigma = active_param[str(bins)][index[0]*2:index[0]*2+2]
            w = nu.random.rand()*100
            u3 =  nu.random.beta(1,1)      
            u1,u2 = nu.random.beta(2,2,2)
            w1 = w*u1; w2 = w*(1-u1)
            mu1 = mu - u2*sigma*nu.sqrt(w2/w1)
            mu2  = mu + u2*sigma*nu.sqrt(w1/w2)
            sigma1 = (u3*(1-u3**2)*sigma**2*w/w1)**(1/2.)
            sigma2 = ((1 - u3)*(1-u2**2)*sigma**2*w/w2)**(1/2.)
            #add split to last 2 indicic's
            for j in range(2):
                active_param[str(temp_bins)][j*2:j*2+2] = eval('mu%i,sigma%i'%(j+1,j+1))
            #det(jocobian)=w_mu*w_sigma and birth percentage)
            critera = abs(1/(mu * sigma) * birth_rate)

            #copy other vars
            j = 2
            while j< temp_bins:
                i = nu.random.randint(0,bins)
                while i  in index:
                    i = nu.random.randint(0,bins)
                index.append(i)
                active_param[str(temp_bins)][j*2] = nu.copy(active_param[str(bins)][i*2] )
                active_param[str(temp_bins)][j*2+1] = abs(active_param[str(bins)][i*2+1] )
                j+=1
            
            #check to see if in bounds
            #other methods of creating params
        elif bins - 1 >= self._min_order:
            #death
            attempt = True #so program knows to attempt a new model
            #choose param randomly delete
            temp_bins = bins - 1 #nu.random.randint(self._min_order,bins)
            critera = 2**(temp_bins) * (1 - birth_rate )
            if .5>nu.random.rand():
                #merge 2 params together
                #W1*u+W2*(1-u) = w 
                u,index = nu.random.rand(),nu.random.randint(0,bins,2)
                #combine the param
                active_param[str(temp_bins)][0] = (
                    active_param[str(bins)][index[0]*2] *(1-u) +
                    active_param[str(bins)][index[1]*2]*u)
                active_param[str(temp_bins)][1] = (
                    active_param[str(bins)][index[0]*2+1] *(1-u) +
                    active_param[str(bins)][index[1]*2+1]*u)
                j = 1
                #copy the rest
                for i in range(bins):
                    if i in index:
                        continue
                    elif j >= temp_bins:
                        break
                    active_param[str(temp_bins)][j*2] = active_param[str(bins)][i*2] 
                    active_param[str(temp_bins)][j*2+1] = abs(active_param[str(bins)][i*2+1] )
                    j+=1
            else:
                #drop random param
                index = nu.random.randint(bins)
                j = 0
                for i in range(bins):
                    if i == index:
                        continue
                    active_param[str(temp_bins)][2*j:2*j+2] = nu.copy(active_param[str(bins)][2*i:2*i+2])
                    j += 1

        else:
            #do nothing
            attempt = False
        if not attempt:
            temp_bins, critera = None,None
        else:
            #do quick llsq to help get to better place
            f_temp = lambda x :(-self.lik(x) - self.prior(x))
            active_param[str(temp_bins)] = fmin_powell(f_temp, 
                                                  active_param[str(temp_bins)],maxfun=10,disp=False)
        return  active_param,temp_bins, attempt, critera
        


    def bic(self):
        '''() -> ndarray
        Calculates the Bayesian Information Criteria for all models in 
        range.
        '''
        seterr('ignore')
        temp = []
        #set minimization funtion to work with powell method
        f_temp = lambda x :-self.lik(x)
        #find maximum likelihood for all models
        for i in range(self._min_order,self._max_order):
            #gen params
            param = self.initalize_param(i)[0]
            param = fmin_powell(f_temp, param ,disp=False)
            temp.append([i,nu.copy(param),self.lik(param)])

        #calculate bic
        #BIC, number of gausians
        out = nu.zeros((len(temp),2))
        for i,j in enumerate(temp):
            out[i,1] = j[2] + j[0]**2*nu.log(self.data.shape[0])
            out[i,0] = j[0]

        return out
    
    def disp_model(self,param):
        '''makes x,y axis for plotting of params'''
        k = len(param)
        #n_points = self.data.shape[0]
        #temp_points = []
        y = nu.zeros_like(self.data[:,1])
        x = self.data[:,0]
        for i in range(0,k,2):
            y += stat_dist.norm.pdf(x,param[i],nu.abs(param[i+1]))
        y *= 1000
        N = nu.sum(y * self.data[:,1])/nu.sum(y**2)
        return x, N * y

if __name__ == '__main__':
    #test to see if working
    import Age_RJMCMC as rj
    import Age_hybrid as hy
    comm = mpi.COMM_WORLD
    rank = mpi.COMM_WORLD.rank
    if rank == 0:
        #send data
        #data = two_dim_image_w_sources((100,100))
        #data.make_param(1)
        print data.param
        
    else:
        #recive data
        data = None
    data = comm.bcast(data,0)
    #print mpi.Get_processor_name(), rank,data[0,0]
    #start rjmcmc
    fun = MCMC_POOL(data)
    top = hy.Topologies('single')
    try:
        a= rj.RJMC_main(fun,top,burnin=500)
    except KeyboardInterrupt:
        if rank == 0:
            print 'age time per iteration is %f' %nu.mean(fun.t)
    if rank == 0:
        import cPickle as pik
        pik.dump((data,a),open('finished.pik','w'),2)
    '''param,sigma = fun.initalize_param(1)
    
    for i in range(100):
        param = fun.proposal(param,sigma)
        print fun.lik({'1':param},'1'),param
    '''
