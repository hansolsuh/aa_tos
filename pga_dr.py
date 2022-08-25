import numpy as np
from datetime import datetime
from numpy.linalg import norm
from numpy import dot
from scipy import optimize, linalg, sparse
import scipy as sp
import sparse_nuclear_norm
import matplotlib.pyplot as plt
from tos import n_tos,atos, aa_tos, Trace

def loss(x):
    return f(x) + G1(x) + G2(x)

class d_r:
    def __init__(self,prox_g,prox_h,it,tol, delta, alpha, beta):
        self.prox_g = prox_g
        self.prox_h = prox_h
        self.it = it
        self.tol = tol
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
    def __call__(self,w,gamma):
        #solving f(x) + g(x) with initialized x0
        n =  w.size
        y = np.zeros(n)
        #y0 is zero
       # y = self.prox_h(2*x,gamma) - x
        step1 = (self.alpha*gamma)/(self.alpha*gamma+self.delta)
        step11 = self.delta/(self.alpha*gamma+self.delta)
        step2 = (self.beta*gamma)/(self.beta*gamma+self.delta)
        step22 = self.delta/(self.beta*gamma+self.delta)
        x = self.prox_g(step1*y+step11*w, step1*self.delta)
        y = y + self.prox_h(step2*(2*x-y)+step22*w, step2*self.delta)-x

        for i in range(self.it):
            x_old = x
            x = self.prox_g(step1*y+step11*w, step1*self.delta)
            y = y + self.prox_h(step2*(2*x-y)+step22*w, step2*self.delta)-x
            gap = norm(x-x_old)
            if gap <= self.tol:
                print('gap break at it :', i)
                break
        return x

class pga:
    def __init__(self,f, solve,it,tol):
        self.f = f
        self.solve = solve
        self.it = it
        self.tol = tol
    def __call__(self,x, step_size):
        fx, grad_fk = self.f.f_grad(x)
        x = self.solve(x-step_size*grad_fk, step_size)
        for i in range(self.it):
            x_old = x
            x = self.solve(x-step_size*grad_fk, step_size)
            gap = norm(x-x_old)
            if gap <= self.tol:
                print("pga tol break at it :", i)
                break
        return x

