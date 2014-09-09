#

#does tests and makes sure correctness of recovering a systematic offset
#from models

import numpy as nu
import pylab as lab
from thuso_quick_fits import quick_cov_MCMC
#initialize real params
#number of points
n=5000
real_param = nu.random.rand(2)*5
real_noise = nu.random.rand()*8.
#####models############
def line(x, param):
    return param[0] * x + param[1]

def line_shift(x,param):
    return param[0] * x +param[1] + 30*nu.sin(nu.pi*x)
#####systematics########
def sine(x,param):
    return param[0] * nu.sin(param[1] * x)
def norm(x,param):
    return (2*nu.pi)**-.5 * param[1]**-1 * nu.exp(
        -((x - param[0])**2/(2* param[1]**2.)))

#create x axis
x = nu.linspace(0,100,n)
y = line(x,real_param) + nu.random.randn(len(x)) * real_noise 
constraints = [[0,5.],[0.,5.]]
chi,param,parambest,chibest = quick_cov_MCMC(x,y,real_param,line,constraints,quiet=True)
fig =lab.figure()
plt1= fig.add_subplot(211)
plt1.hist(y-line(x,nu.mean(param,0)),nu.int(n**.5))
plt1.set_title('no systematic, just noise')

'''y_shift = line_shift(x,real_param) + nu.random.randn(len(x)) * real_noise 
chi_shift,param_shift,parambest,chibest = quick_cov_MCMC(x,y_shift,real_param,line_shift,constraints,quiet=True)
plt2= fig.add_subplot(212)
plt2.hist(y_shift-line(x,nu.mean(param_shift,0)),nu.int(n**.5))
plt2.set_title("with systematic sin")
fig.canvas.draw()

'''
y_err = (line(x,real_param) + nu.random.randn(len(x)) * real_noise +
         nu.random.randn(len(x)) * 4)
chi_err,param_err,parambest,chibest=quick_cov_MCMC(x,y_err,real_param,line_shift,constraints,quiet=True)
plt3= fig.add_subplot(212)
plt3.hist(y_err-line(x,nu.mean(param_err,0)),nu.int(n**.5))
plt3.set_title("with 2 error componets")
fig.canvas.draw()
