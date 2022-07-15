import csv
from scipy.sparse import csr_matrix
from numpy.linalg import norm
import numpy as np

def generate_data():
    #Tomography Data
    columns = []
    rows    = []
    values  = []
   
    with open('tomo/Fmatrix.txt', newline = '\n') as matdata:
        Fmat_raw = csv.reader(matdata, delimiter='\t')
        for row in Fmat_raw:
            columns.append(int(row[0])-1)
            rows.append(int(row[1])-1)
            values.append(float(row[2]))
       
    F = csr_matrix((values,(rows, columns)), shape=(1440,2500))
    b = np.loadtxt('tomo/bvec.txt')
    GT = np.loadtxt('tomo/GTvec.txt')

    return F,b,GT

class f1_copt:
    def __init__(self, e, lbd):
        self.e = e
        self.lbd = lbd
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def prox(self,x, step_size):
        step_size = np.real(step_size)
        out = x - step_size*self.lbd*self.e
        return np.ravel(out)
    def f_grad(self,x,return_gradient=True):
        fval = self.lbd*np.dot(self.e,x)
        if not return_gradient:
            return fval
        out = self.lbd*self.e
        out = np.ravel(out)
        return fval, out

class f2_copt:
    def __init__(self, mu):
        self.mu = mu
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def prox(self,x, step_size):
        step_size = np.real(step_size)
        out = (x + np.sqrt(x**2 + 4*self.mu*step_size))/2
        return np.ravel(out)
    def f_grad(self,x,return_gradient=True):
        fval = -self.mu*np.sum(np.log(x))
        if not return_gradient:
            return fval
        out = -self.mu/x
        out = np.ravel(out)
        return fval, out

class f3_copt:
    def __init__(self, F,  b, FTb):
        self.F = F
        self.b = b
        self.FTb = FTb
    def __call__(self,x):
        return self.f_grad(x,return_gradient=False)
    def f_grad(self,x, return_gradient=True):
        Fx = self.F @ x
        fval = 0.5*norm(Fx-self.b)**2
        if not return_gradient:
            return fval
        FTFx = self.F.T @ Fx
        out = np.ravel(FTFx - self.FTb)
        return fval, out
