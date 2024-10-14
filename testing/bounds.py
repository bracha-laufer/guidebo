import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

def h2(y):
    return (1+y)*np.log(1+y) - y

### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.minimum(mu,x),mu)

def hoeffding_minus(mu, x, n):
    return -n * h1(np.maximum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.ceil(n*x),n,mu),1e-10))+1

def binom_minus(mu, x, n):
    return np.log(max(1-binom.cdf(np.floor(n*x),n,mu),1e-10))

def binom_plus(mu, x, n):
    return np.log(max(binom.cdf(np.ceil(n*x),n,mu),1e-10))

def bentkus_minus(mu, x, n):
    return np.log(max(1-binom.cdf(np.floor(n*x),n,mu),1e-10))+1    

def hb_p_value_left_tail(r_hat,n,alpha):
    bentkus_p_value = np.e * binom.cdf(np.ceil(n*r_hat),n,alpha)
    def h1(y,mu):
        with np.errstate(all='ignore'):
            return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
    hoeffding_p_value = np.exp(-n*h1(np.minimum(r_hat,alpha),alpha))
    return np.fmin(bentkus_p_value,hoeffding_p_value)

def hb_p_value_right_tail(r_hat,n,alpha):
    bentkus_p_value = np.e * (1-binom.cdf(np.floor(n*r_hat),n,alpha))
    def h1(y,mu):
        with np.errstate(all='ignore'):
            return y * np.log(y/mu) + (1-y) * np.log((1-y)/(1-mu))
    hoeffding_p_value = np.exp(-n*h1(np.maximum(r_hat,alpha),alpha))
    return np.fmin(bentkus_p_value,hoeffding_p_value)    

def binom_p_value_left_tail(r_hat,n,alpha):
    binom_p_value = binom.cdf(np.ceil(n*r_hat),n,alpha)
    return binom_p_value 

def binom_p_value_right_tail(r_hat,n,alpha):
    binom_p_value = 1-binom.cdf(np.floor(n*r_hat),n,alpha)
    return binom_p_value       

def HB_mu_plus(muhat, n, delta, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n) 
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)

def HB_mu_minus(muhat, n, delta, maxiters):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_minus(mu, muhat, n) 
        bentkus_mu = bentkus_minus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1e-10) > 0:
        return 0
    else:
        return brentq(_tailprob, 1e-10, muhat, maxiter=maxiters)

def mu_plus(muhat, n, delta, maxiters, method = 'bentkus'):
    dict_plus = {'bentkus':bentkus_plus, 'hoeffding':hoeffding_plus, 'binom':binom_plus}
    func_plus = dict_plus[method]
    def _tailprob(mu):
        return func_plus(mu, muhat, n) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)        

def mu_minus(muhat, n, delta, maxiters, method = 'bentkus'):
    dict_minus = {'bentkus':bentkus_minus, 'hoeffding':hoeffding_minus, 'binom':binom_minus}
    func_minus = dict_minus[method]
    def _tailprob(mu):
        return func_minus(mu, muhat, n) - np.log(delta)
    if _tailprob(1e-10) > 0:
        return 0
    else:
        return brentq(_tailprob, 1e-10, muhat, maxiter=maxiters)        

if __name__ == "__main__":
    delta = 0.1
    ncalib = 5000
    rhat = np.linspace(0.01,0.05,100)
    num_grid_hbb = 200
    epsilon = 1e-10 
    maxiters = int(1e5)
    rplus, rminus = np.zeros(rhat.shape), np.zeros(rhat.shape)
    for i,r in enumerate(rhat):
        rplus[i] = HB_mu_plus(r, ncalib, delta, maxiters=maxiters)
        rminus[i] = HB_mu_minus(r, ncalib, delta, maxiters=maxiters)


    
    plt.figure()
    plt.plot(rhat,rhat)
    plt.plot(rhat,rplus)
    plt.plot(rhat,rminus)
    plt.savefig('bouns.jpg')

    
    methods = ['bentkus','hoeffding','binom']
    rplus = {m: np.zeros(rhat.shape) for m in methods}
    rminus = {m: np.zeros(rhat.shape) for m in methods}

    plt.figure()
    plt.plot(rhat,rhat)

    colors = 'rbg'

    for j, method in enumerate(methods):
        for i,r in enumerate(rhat):
            rplus[method][i] = mu_plus(r, ncalib, delta, maxiters=maxiters, method=method)
            rminus[method][i] = mu_minus(r, ncalib, delta, maxiters=maxiters, method=method)
    
        plt.plot(rhat,rplus[method],c=colors[j],label=method)
        plt.plot(rhat,rminus[method],c=colors[j])
    
    plt.legend()
    plt.savefig('compare_bouns.jpg')

