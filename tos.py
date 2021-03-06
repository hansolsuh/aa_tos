import numpy as np
from datetime import datetime
from numpy.linalg import norm
from numpy import dot
from scipy import optimize, linalg, sparse
import scipy as sp


#Copy of ATOS from copt by Pedregosa et al, Zenodo doi...
def atos(
    f_grad,
    x0,
    prox_1,
    prox_2,
    step_size,
    callback=None,
    tol=1.e-6,
    max_iter=1000,
    line_search=True,
    backtracking_factor=0.7,
    max_iter_backtracking=100,
    h_Lipschitz=None,
    barrier=None,
    inner_aa=None,
    outer_aa=None
):
    success = False
    if inner_aa is None:
        inner_aa = 0
    if outer_aa is None:
        outer_aa = 0
    
    z_old = x0
    z = prox_2(x0, step_size )
    fk,grad_fk = f_grad(z)
    y = z-step_size*grad_fk
    x = prox_1(y, step_size )
    u = np.zeros_like(x)

    if inner_aa !=0:
        p = y
        p_aa   = []
        Q_aa   = []

    if outer_aa !=0:
        #Safeguard from Fu,Zhang,Boyd, 2020
        safeguard = True
        n_aa      = 0
        iter_aa   = 0 #R_aa in Fu paper
        D_aa      = 1.e7 #TODO choose?
        R_safe    = 10
        r         = (x-z)/step_size #TODO making up u0
        r0_norm = norm(r)
        eps_aa = 1.e-6
        R_aa = []
        u_aa = []

    for it in range(max_iter):
        aa_mk_inner = min(inner_aa,it)
        aa_mk_outer = min(outer_aa,it)

        grad_fk_old = grad_fk
        fk,grad_fk = f_grad(z)

        if inner_aa != 0:
            p_old = p
            p = z-step_size*(u+grad_fk)
            q = p - y
            len_Q = len(Q_aa)
            if len_Q >= aa_mk_inner and len_Q != 0:
                Q_aa.pop(0)
            Q_aa.append(q)
            Q_sol = AA_LQ(Q_aa,1.e-6)

            if len(p_aa) >= aa_mk_inner and len(p_aa) != 0:
                p_aa.pop(0)
            p_aa.append(p)
            y_test = sum([p_aa[i]*val for i,val in enumerate(Q_sol)])

        if inner_aa == 0:
            x = prox_1(z- step_size*(u+grad_fk), step_size)
        else:
            x_test = prox_1(y_test, step_size)
            fx_test = f_grad(x_test, return_gradient=False) 
            fx,grad_fk = f_grad(x)
            incr = x-z
            f_if_test = fx_test - fx - grad_fk.dot(incr) - (norm(incr)**2)/(2*step_size)
            if f_if_test <= 0:
                x = x_test
            else:
                x = prox_1(p, step_size)
                y = p

        incr = x-z
        norm_incr = np.linalg.norm(incr)
        
        fx = f_grad(x, return_gradient=False)
        ls = norm_incr > 1e-7 and line_search
        if ls:
            for it_ls in range(max_iter_backtracking):
                rhs = fk+grad_fk.dot(incr) + (norm_incr **2)/(2*step_size)
                ls_tol = fx-rhs
                if ls_tol <= 1.e-12:
                    break
                else:
                    step_size *= backtracking_factor

        z = prox_2(x+step_size*u, step_size )
        u_old = u
        u += (x-z)/step_size
        if it ==0:
            r0_norm = norm(u-u_old)
        #Doing outer on u only
        if outer_aa != 0:
            r = u - u_old
            len_R = len(R_aa)
            if len_R >= aa_mk_outer and len_R !=0:
                R_aa.pop(0)
            R_aa.append(r)
            R_sol = AA_LQ(R_aa,1.e-6)

            if len(u_aa) >= aa_mk_outer and len(u_aa) != 0:
                u_aa.pop(0)
            u_aa.append(u)
            u_test = sum([u_aa[i]*val for i,val in enumerate(R_sol)])

            #Safeguarding
            if safeguard or iter_aa >= R_safe:
                if norm(r) <= D_aa*r0_norm*(n_aa/R_safe + 1)**(-1-eps_aa):
                    u = u_test
                    n_aa = n_aa + 1
                    safeguard = False
                    R_safe = 1
                else:
                    #z = z_TOS
                    R_safe = 0
            else:
                u = u_test
                n_aa = n_aa + 1
                R_safe = R_safe + 1

        certificate = norm_incr / step_size
        if ls and h_Lipschitz is not None:
            if h_Lipschitz == 0:
                step_size = step_size * 1.02
            else:
                quot = h_Lipschitz ** 2
                tmp = np.sqrt(step_size ** 2 + (2 * step_size / quot) * (-ls_tol))
                step_size = min(tmp, step_size * 1.02)

        if callback is not None:
            if callback(locals()) is False:
                break

        if it > 0 and certificate < tol:
            if barrier != None:
                if barrier > 1.e-12:
                    barrier /= 1.1
                else:
                    success = True
                    break
            else:                
                success = True
                break

    return optimize.OptimizeResult(
        x=x, success=success, nit=it, certificate=certificate, step_size=step_size
    )

def aa_tos(
    f_grad,
    z0,
    prox_1,
    prox_2,
    step_size,
    callback=None,
    tol=1.e-6,
    max_iter=1000,
    line_search=True,
    barrier=None,
    inner_aa=None,
    outer_aa=None
):
    success = False
    if inner_aa is None:
        inner_aa = 0
    if outer_aa is None:
        outer_aa = 0

    w = prox_2(z0, step_size )
    fk,grad_fk = f_grad(w)
    y = 2*w-z0-step_size*grad_fk
    x = prox_1(y,step_size)
    z = z0 + x - w

    if inner_aa !=0:
        p = y
        p_aa   = []
        Q_aa   = []

    if outer_aa !=0:
        #Safeguard from Fu,Zhang,Boyd, 2020
        safeguard = True
        n_aa      = 0
        iter_aa   = 0 #R_aa in Fu paper
        D_aa      = 1.e7 #TODO choose?
        R_safe    = 10
        r    = z - z0
        r0_norm = norm(r)
        eps_aa = 1.e-6
        R_aa = []
        z_aa = []

    for it in range(max_iter):
        aa_mk_inner = min(inner_aa,it)
        aa_mk_outer = min(outer_aa,it)
        
        grad_fk_old = grad_fk
        w = prox_2(z, step_size)
        fk,grad_fk = f_grad(w)
        if inner_aa != 0:
            p_old = p
            p = 2*w-z-step_size*grad_fk
            q = p - y
            len_Q = len(Q_aa)
            if len_Q >= aa_mk_inner and len_Q != 0:
                Q_aa.pop(0)
            Q_aa.append(q)
            Q_sol = AA_LQ(Q_aa,1.e-6)

            if len(p_aa) >= aa_mk_inner and len(p_aa) != 0:
                p_aa.pop(0)
            p_aa.append(p)
            y_test = sum([p_aa[i]*val for i,val in enumerate(Q_sol)])

        if inner_aa == 0:
            x = prox_1(2*w-z-step_size*grad_fk, step_size)
        else:
            x_test  = prox_1(y_test, step_size)
            fx_test = f_grad(x_test, return_gradient=False) 
            fx,grad_fk = f_grad(x)
            incr = w-x
            f_if_test = fx_test - fx - grad_fk.dot(incr) - (norm(incr)**2)/(2*step_size)
            if f_if_test <= 0:
                x = x_test
            else:
                x = prox_1(p, step_size)
                y = p

        z_old = z
        z = z + x - w

        if outer_aa != 0:
            r = z - z_old
            len_R = len(R_aa)
            if len_R >= aa_mk_outer and len_R !=0:
                R_aa.pop(0)
            R_aa.append(r)
            R_sol = AA_LQ(R_aa,1.e-6)

            if len(z_aa) >= aa_mk_outer and len(z_aa) != 0:
                z_aa.pop(0)
            z_aa.append(z)
            z_test = sum([z_aa[i]*val for i,val in enumerate(R_sol)])

            #Safeguarding
            if safeguard or iter_aa >= R_safe:
                if norm(r) <= D_aa*r0_norm*(n_aa/R_safe + 1)**(-1-eps_aa):
                    z = z_test
                    n_aa = n_aa + 1
                    safeguard = False
                    R_safe = 1
                else:
                    #z = z_TOS
                    R_safe = 0
            else:
                z = z_test
                n_aa = n_aa + 1
                R_safe = R_safe + 1



        certificate = (1/step_size)*norm(x-w)/(1+norm(z))

        if callback is not None:
            if callback(locals()) is False:
                break

        if it > 0 and certificate < tol:
            if barrier != None:
                if barrier > 1.e-12:
                    barrier /= 1.1
                else:
                    success = True
                    break
            else:                
                success = True
                break

    return optimize.OptimizeResult(
        x=x, success=success, nit=it, certificate=certificate, step_size=step_size
    )

def AA_LQ(R,lbd):
    RTR = np.matmul(np.array(R),np.array(R).T)
    one_R = np.ones(len(R))
    np.fill_diagonal(RTR,RTR.diagonal()+lbd)
    R_sol = np.linalg.solve(RTR,one_R)
    R_sol = R_sol/sum(R_sol)
    return R_sol

class Trace:
    def __init__(self, f=None, freq=1):
        self.trace_x = []
        self.trace_time = []
        self.trace_fx = []
        self.trace_step_size = []
        self.start = datetime.now()
        self._counter = 0
        self.freq = int(freq)
        self.f = f

    def __call__(self, dl):
        if self._counter % self.freq == 0:
            if self.f is not None:
                self.trace_fx.append(self.f(dl["x"]))
            else:
                self.trace_x.append(dl["x"].copy())
            delta = (datetime.now() - self.start).total_seconds()
            self.trace_time.append(delta)
            self.trace_step_size.append(dl["step_size"])
            self.trace_fx.append(dl["fk"])

        self._counter += 1

