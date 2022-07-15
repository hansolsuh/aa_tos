import csv
from scipy.sparse import csr_matrix
from scipy import optimize
from numpy.linalg import norm
import numpy as np

class simplex_bisect:
    def __init__(self,x,H):
        self.H = H
        self.x = x
    def __call__(self,t):
        out = sum(np.maximum(self.x-self.H*t,0)) - 1
        return out

def generate_data():
    #Matlab Portfolio Opt Data
    Sigma = np.loadtxt('dydata/sigma.txt')            
    m = np.loadtxt('dydata/m.txt')
    r = np.loadtxt('dydata/r.txt')
    gt = np.loadtxt('dydata/x_cvx_true.txt')
    eval1 = np.loadtxt('dydata/Sigma_eval.txt')
    return Sigma,r,m,gt,eval1

class f1_copt:
    def __init__(self,Q):
        self.Q = Q
    def __call__(self,x):
        Qx = self.Q @ x
        fval = 0.5* x.T @ Qx
        return fval
    def f_grad(self,x,return_gradient=True):
        Qx   = self.Q @ x
        fval = 0.5* x.T @ Qx
        if not return_gradient:
            return fval
        out = np.ravel(Qx)
        return fval,out

#Unit Simplex Projection
class f2_copt:
    def __init__(self):
        pass
    def __call__(self,x):
        return np.sum(x)
    def prox(self,x, step_size):
#        if isinstance(step_size,np.ndarray):
#            lb = np.max(x/step_size-1)
#            ub = np.max(x/step_size)
#            bss = simplex_bisect(x,step_size)
#            root = optimize.bisect(bss,lb,ub)
#            out = np.maximum(x-step_size*root,0)
#        else:
        n = x.size
        idx_sort = np.argsort(x)[::-1]
        y = x[idx_sort]
        cumsum_list = (np.cumsum(y)-1)
        temp        = np.arange(1,n+1)
        cumsum_list = cumsum_list/temp
        temp        = cumsum_list[np.sum(y>cumsum_list)-1]
        out         = np.maximum(x-temp,0)
        return np.ravel(out)

#Project Half Space 
#For <m,x> >= r
class f3_copt:
    def __init__(self, r,m):
        self.r = r
        self.m = m
    def prox(self,x, step_size):
        mx = np.dot(self.m,x)
        if mx <= -self.r:
            return np.copy(x)

        nn = np.dot(self.m, self.m)
        #out = x - (1/step_size)*(np.maximum(mx + self.r,0)/nn)*self.m 
        out = x - (np.maximum(mx + self.r,0)/nn)*self.m 
        return np.ravel(out)

