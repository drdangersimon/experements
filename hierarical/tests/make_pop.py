import likelihood_class as lik
import scipy.stats as stats
import pylab as lab
from MC_utils import list_dict_to
import triangle as tri
#from Age_RJMCMC import RJMC_main
from Age_mltry import multi_main
import mpi_top as hy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as nu
try:
    import daft
except:
    pass
'''Toy examples of hieriacal model fitting using RJMCMC'''


'''Linear regression of population of lines'''
'''Want to estimate mean and std (M,E) of a varlible from a family of Linear
regression lines. Each line is modeled by y_i = m_i*x+b+e where e ~N(0,2)
posteror should look like:
P(M,E|m_i,y_i) = P(m_i,y_i|M,E)P(M)P(E) = P(m_i|y_i)P(m_i|E,M)P(E)P(M)P(b)
P(m_i|y_i) = \prod_{i=1}^N N(y_i-m_i,e)
P(m_i|E,M) = \prd_{K} N(M-sum(m_i)/N,E)
prors
P(E) = U(1,5)
P(M) = U(-10,10)
P(b) = U(-1,1)
P(m_i|N,E) = N(M,E)

'''


def make_data(M=stats.uniform(-10, 20), E=stats.uniform(1, 4), k=1, N=50):
    '''makes data drawing m_i from M, E'''
    # make true params
    # M,E
    true_top_param = [M.rvs(), E.rvs()]
    # m_i,b
    true_bottom_param = []
    data = {}
    for i in xrange(k):
        # Make param from M, E
        true_bottom_param.append([stats.norm.rvs(true_top_param[0],
                                                 true_top_param[1]),
                                    stats.uniform.rvs(-1, 2)])
        # generate data
        data[i] = generate_data(N, true_bottom_param[-1])
    return data, true_top_param, true_bottom_param


def generate_data(Npoints, param):
    '''Makes random data'''
    data = nu.zeros((Npoints, 2))
    data[:, 0] = nu.sort(nu.random.rand(Npoints)*2)
    data[:, 1] = param[0]*data[:, 0] + param[1]
    data[:, 1] += nu.random.randn(Npoints) * 2
    return data


def lower_model(x, m, b):
    '''makes lower model for fitting'''
    return m * x + b


def upper_model(*args):
    '''makes upper model for fitting'''
    return nu.mean(args), nu.std(args)


class hieriacal(lik.Example_lik_class):
    def __doc__(self):
        '''Likelihood function for hieracial linear fitting
        Trying to figure out:
        P(M|E,m_i,b_i,y_j) = P(M)P(E)P(b)P(y|m_i,b)P(m_i|M,E)
        P(y|m_i,b) is likelhood
        rest are priors
        '''
    def __init__(self, data):
        self.data = data
        k = len(data.keys())
        self.models = {k: []}

    def lik(self, param, bins):
        '''calc likelihood P(y|m_i,b)'''
        tdata = nu.zeros_like(self.data[self.data.keys()[0]])
        # lower
        for i in self.data.keys():
            #yield  1. ,i
            tdata[:, 0] = self.data[i][:, 0]
            tdata[:, 1] = lower_model(tdata[:, 0], param[bins][i][3],
                                      param[bins][i][2])
            yield stats.norm.logpdf(self.data[i][:, 1],
                                         tdata[:, 1], 2).sum(), i
        

    def initalize_param(self, bins):
        '''
        P(E) = U(1,5)
        P(M) = U(-10,10)
        P(b) = U(-1,1)
        P(m_i|M,E) = N(M,E)'''
        # make new guess with each bin
        param = {}
        for i in xrange(bins):
            # M, E, b, m_i
            param[i] = [stats.uniform.rvs(-10, 20), stats.uniform.rvs(1, 4),
                        stats.uniform.rvs(-1, 2), 0.]
            # calculate m_i
            param[i][-1] = stats.norm.rvs(param[i][0], param[i][1])

        return param, nu.eye(bins*4)
    
    def prior(self, param, bins):
        '''
        P(E) = U(1,5)
        P(M) = U(-10,10)
        P(b) = U(-1,1)
        P(m_i|M,E) = N(M,E)'''
        # M,E,b,m_i
        for i in self.data.keys():
            out_lik = 0.
            if i == 0:
                # P(E)
                out_lik += stats.uniform.logpdf(param[bins][i][1], 1, 4)
                # P(M)
                out_lik += stats.uniform.logpdf(param[bins][i][0], -10, 20)
            # P(b)
            out_lik += stats.uniform.logpdf(param[bins][i][2], -1, 2)
            # P(m_i|M,E)
            out_lik += stats.norm.logpdf(param[bins][i][3], param[bins][i][0],
                                         param[bins][i][1])
            yield out_lik,i
    
    def model_prior(self, model):
        return 0.
    
    def proposal(self, Mu, Sigma):
        # get out of dict
        mu = nu.hstack(Mu.values())
        temp =  nu.random.multivariate_normal(mu, Sigma)
        out = Mu.copy()
        temp = nu.reshape(temp,(len(Mu.keys()), 4))
        for i in out.keys():
            out[i] = temp[i]
        return out
        
    def step_func(self, step_crit, param, step_size, model):
        if step_crit > .30 and nu.any(step_size[model].diagonal() < 10**2):
            step_size[model] *= 1.05
        elif step_crit < .15 and nu.any(step_size[model].diagonal() > 10**-4):
            step_size[model] /= 1.05
        # cov matrix
        if len(param) % 200 == 0 and len(param) > 0.:
            t = len(param)/10**5.
            #take only unique params
            # convert from dict to ndarray
            temp = nu.cov(list_dict_to(param[-200:], param[0].keys()).T)
            # make sure not stuck
            # if nu.any(temp.diagonal() > 10**-6):
            step_size[model] = (1-t) * temp + t * step_size[model]
        return step_size[model]
    
    def birth_death(self, birth_rate, bins, param):
        return param, bins ,False, 0

class no_pooling(hieriacal):
    '''Fits models independandly from hierarical parameters. Gets hierarical
    parameters at the end.
    '''
    def proposal(self, Mu, Sigma):
        # get out of dict
        mu = nu.hstack(Mu.values())
        # pdb.set_trace()
        temp =  nu.random.multivariate_normal(mu, Sigma)
        out = Mu.copy()
        temp = nu.reshape(temp,(len(Mu.keys()), 4))
        m_i = []
        for i in out.keys():
            out[i] = temp[i]
            m_i.append(temp[i][3])
        temp[0][0] = nu.mean(m_i)
        temp[0][1] =  nu.std(m_i)
        return out

    def prior(self, param, bins):
        '''P(E) = U(1,5)
        P(M) = U(-10,10)
        P(b) = U(-1,1)
        P(m_i|N,E) = N(M,E)'''
        #M,E,b,m_i
        for i in self.data.keys():
            out_lik = 0.
            #P(E)
            out_lik += stats.uniform.logpdf(param[bins][i][1],1,4)
            #P(M)
            out_lik += stats.uniform.logpdf(param[bins][i][0],-10,20)
            #P(b)
            out_lik += stats.uniform.logpdf(param[bins][i][2],-1,2)
            #P(m_i)
            out_lik += stats.uniform.logpdf(param[bins][i][3],-15,50)
            #P(m_i|M,E)
            '''out_lik += stats.norm.logpdf(param[bins][i][3],param[bins][i][0],
                                         param[bins][i][1])'''

            yield out_lik, i
class complete_pooling(hieriacal):
    '''Takes all data and makes 1 fit for the global pop
    '''
    def __init__(self, data):
        #make into 1 big data file
        self.data = {}
        x,y = [] ,[]
        for key in data.keys():
            x.append(data[key][:,0])
            y.append(data[key][:,1])
        self.data[0] = nu.vstack((nu.ravel(x),nu.ravel(y))).T
        k = len(data.keys())
        self.models = {k: []}

    def lik(self, param, bins):
        '''calc likelihood P(y|m_i,b)'''
        tdata = nu.zeros_like(self.data[self.data.keys()[0]])
        # lower
        for i in self.data.keys():
            #yield  1. ,i
            tdata[:, 0] = self.data[i][:, 0]
            tdata[:, 1] = lower_model(tdata[:, 0], param[bins][i][0],
                                      param[bins][i][2])
            yield stats.norm.logpdf(self.data[i][:, 1],
                                         tdata[:, 1], 2).sum(), i
        


    def prior(self,  param, bins):
        '''P(M) ~ U(-10,10)
        P(b) ~ U(-1,1)'''
        for i in self.data.keys():
            out_lik = 0.
            out_lik += stats.uniform.logpdf(param[bins][i][2], -1, 2)
            out_lik += stats.uniform.logpdf(param[bins][i][0], -10, 20)
            yield out_lik, i
    
def plot_data(data):
    '''plots data'''
    assert  isinstance(data,dict), 'Must input a dictionary'
    fig = lab.figure()
    plt = fig.add_subplot(111)
    
    for i in data.keys():
        plt.plot(data[i][:,0],data[i][:,1],label=str(i))
    return fig

def plot_param(param,Data):
    '''plots parameter'''
    data = {}
    if not isinstance(param,dict):
        #make into dict
        Param = {}

    else:
        Param = param.copy()
    #make data
    for i in Param.keys():
        #M,E,b,m_i
        data[i] = nu.vstack((Data[i][:,0],lower_model(Data[i][:,0], Param[i][3], Param[i][2]))).T
    #plot 
    return data,plot_data(data)

def plot_pgm_hierarical():
    '''Plots PGM model for hierarical plot'''
    pgm = daft.PGM([3,4],origin=[-1, -1])
    # add hierarical params
    pgm.add_node(daft.Node('Mean','M',1,2))
    pgm.add_node(daft.Node('Sigma',r'$\Sigma$',0,2))
    # Laten Varibles
    pgm.add_node(daft.Node('local slope',r'$m_i$',0,1))
    pgm.add_node(daft.Node('local intercept', r'$b_i$',1,1))
    # data
    pgm.add_node(daft.Node('data',r'$y_j$',0.5,0))

    # adding and edge
    pgm.add_edge('Mean','local slope')
    pgm.add_edge('Sigma','local slope')
    pgm.add_edge('local slope','data')
    pgm.add_edge('local intercept','data')

    # make plates
    pgm.add_plate(daft.Plate([-.4,.5,1.8,.8],label=r'Models$_{i = 0, \cdots, k}$'))
    pgm.add_plate(daft.Plate([-.5,-.5,2.3,2],label=r'Data$_{j = 0, \cdots, N}$ in Model$_i$'))
    pgm.render()
    pgm.figure.savefig("Hierarical_PGM.png", dpi=150)
    
def plot_pgm_non_pooled():
    '''Plots PGM model for hierarical plot'''
    pgm = daft.PGM([3,4],origin=[-1, -1])
    # add hierarical params
    pgm.add_node(daft.Node('Mean','M',1,2))
    pgm.add_node(daft.Node('Sigma',r'$\Sigma$',0,2))
    # Laten Varibles
    pgm.add_node(daft.Node('local slope',r'$m_i$',0,1))
    pgm.add_node(daft.Node('local intercept', r'$b_i$',1,1))
    # data
    pgm.add_node(daft.Node('data',r'$y_j$',0.5,0))

    # adding and edge
    pgm.add_edge('local slope','Mean')
    pgm.add_edge('local slope','Sigma')
    pgm.add_edge('local slope','data')
    pgm.add_edge('local intercept','data')

    # make plates
    pgm.add_plate(daft.Plate([-.4,.5,1.8,.8],label=r'Models$_{i = 0, \cdots, k}$'))
    pgm.add_plate(daft.Plate([-.5,-.5,2.3,2],label=r'Data$_{j = 0, \cdots, N}$ in Model$_i$'))
    pgm.render()
    pgm.figure.savefig("Non_pooled_PGM.png", dpi=150)

    
def plot_pgm_pooled():
    '''Plots PGM model for hierarical plot'''
    pgm = daft.PGM([3,4],origin=[-1, -1])
    
    # Laten Varible
    pgm.add_node(daft.Node('global intercept', r'$b$',0,2))
    pgm.add_node(daft.Node('Mean',r'$M$',1,2))
    # data
    pgm.add_node(daft.Node('data',r'$y_j$',0.5,0))

    # adding and edge
    pgm.add_edge('Mean','data')
    pgm.add_edge('global intercept','data')

    # make plates
    #pgm.add_plate(daft.Plate([-.4,.5,1.8,.8],label=r'Models$_{i = 0, \cdots, k}$'))
    pgm.add_plate(daft.Plate([-.5,-.5,2.3,2],label=r'Data$_{j = 0, \cdots, N}$'))
    pgm.render()
    pgm.figure.savefig('Pooled_PGM.png', dpi=150)

    
def run_paralell(args):
    '''wrapper for parallel run import data and which model to use'''
    data, hier = args
    if hier == 'hier':
        fun,top = hieriacal(data),hy.Topologies('single')
    elif hier == 'no_pool':
        fun,top = no_pooling(data),hy.Topologies('single')
    else:
        fun,top = complete_pooling(data),hy.Topologies('single')

    return multi_main(fun,top,burnin=5000)

if __name__ == '__main__':
    from multiprocessing import Pool
    import cPickle as pik
    pool = Pool(3)
    # Plot PGMs
    try:
        plot_pgm_hierarical()
        plot_pgm_non_pooled()
        plot_pgm_pooled()
    except:
        pass

    #create data
    k = 40
    Data,true_top_param,true_bottom_param = make_data(k=k)
    #run MCMC with RJMC by setting birth rate to 0 in parallel
    out = pool.map(run_paralell,iter([(Data, 'hier'), (Data, 'no_pool'),
                                 (Data, '')]))
    #plot
    data = list_dict_to(out[0].param[k],out[0].param[k][0].keys())
    data1 = list_dict_to(out[1].param[k],out[1].param[k][0].keys())
    data2 = list_dict_to(out[2].param[k],out[2].param[k][0].keys())
    #save data incase of crash
    pik.dump((data,data1,data2,Data,true_top_param,true_bottom_param),
             open('temp.pik','w'),2)
    #calc M and E for non_hierachal
    if k > 1:
        data1[:,0] = data1[:,range(3,data1.shape[1],4)].mean(1)
        data1[:,1] = data1[:,range(3,data1.shape[1],4)].std(1)
    #remove junk hyper params
    true = [true_top_param[0], true_top_param[1]]
    #m_i,b
    true.append(true_bottom_param[0][1])
    true.append(true_bottom_param[0][0])
    index = [0,1,2,3]
    label = ['M','E','b_0','m_0']
    ii = 0
    for j,i in enumerate(xrange(4,data.shape[1],4)):
        index.append(i+2)
        index.append(i+3)
        label.append('b.%i'%(j+1))
        label.append('m.%i'%(j+1))
        true.append(true_bottom_param[j+1][1])
        true.append(true_bottom_param[j+1][0])
        ii+=1
        if ii > 3:
            break
    data = data[:,index]
    data1 = data1[:,index]
    #tringle plot
    
    fig_hi = tri.corner(data[10**4:],labels = label,truths=true)
    fig_no_pool = tri.corner(data1[10**4:],labels = label,truths=true)
    fig_comple_pool = tri.corner(data2[:10**4,[0,2]],labels=[r'$M_{global}$','b']
                                , truths = [true_top_param[0],nu.asarray(true_bottom_param)[:,1].mean()])
    fig_hi.suptitle('Hierarical with %i means'%k, fontsize=16)
    fig_no_pool.suptitle('No pool with %i means'%k,fontsize=16)
    fig_comple_pool.suptitle('Complete pool',fontsize=16)
    lab.show()
