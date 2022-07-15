"""
Group lasso with overlap
========================

Comparison of solvers for a least squares with
overlapping group lasso regularization.

References
----------
This example is modeled after the experiments in `Adaptive Three Operator Splitting <https://arxiv.org/pdf/1804.02339.pdf>`_, Appendix E.3.
"""
import copt as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import linalg as splinalg
from libsvm.svmutil import *

def generate_data():
    np.random.seed(0)
    
    n_samples, n_features = 100, 1002
    
    # .. generate some data ..
    # .. the first set of blocks is
    groups = [np.arange(8 * i, 8 * i + 10) for i in range(125)]
    ground_truth = np.zeros(n_features)
    g = np.random.randint(0, len(groups), 10)
    for i in g:
        ground_truth[groups[i]] = np.random.randn()
    
    A = np.random.randn(n_samples, n_features)
    p = 0.95  # create a matrix with correlations between features
    for i in range(1, n_features):
        A[:, i] = p * A[:, i] + (1 - p) * A[:, i-1]
    A[:, 0] /= np.sqrt(1 - p ** 2)
    A = preprocessing.StandardScaler().fit_transform(A)
    b = A.dot(ground_truth) + np.random.randn(n_samples)
    
    # make labels in {0, 1}
    b = np.sign(b)
    b = (b + 1) // 2
    
    
    # .. compute the step-size ..
    max_iter = 5000
    f = LogLoss(A, b)
    step_size = 1. / f.lipschitz

    return A,b,f,groups,ground_truth,n_features,step_size

def lib_svm_data():
    y,x = svm_read_problem('data/real-sim',return_scipy=True)
    y[y<0] = 0
    [n,m] = x.shape

    groups = [np.arange(8 * i, 8 * i + 10) for i in range(2619)]

    f = LogLoss(x,y)
    step_size = 1. / f.lipschitz
    return x,y,f,groups,0,n,step_size

class LogLoss:
    r"""Logistic loss function.

  The logistic loss function is defined as

  .. math::
      -\frac{1}{n}\sum_{i=1}^n b_i \log(\sigma(\bs{a}_i^T \bs{x}))
         + (1 - b_i) \log(1 - \sigma(\bs{a}_i^T \bs{x}))

  where :math:`\sigma` is the sigmoid function
  :math:`\sigma(t) = 1/(1 + e^{-t})`.

  The input vector b verifies :math:`0 \leq b_i \leq 1`. When it comes from
  class labels, it should have the values 0 or 1.

  References:
    http://fa.bianp.net/blog/2019/evaluate_logistic/
  """

    def __init__(self, A, b, alpha=0.0):
        if A is None:
            A = sparse.eye(b.size, b.size, format="csr")
        self.A = A
        if np.max(b) > 1 or np.min(b) < 0:
            raise ValueError("b can only contain values between 0 and 1 ")
        if not A.shape[0] == b.size:
            raise ValueError("Dimensions of A and b do not coincide")
        self.b = b
        self.alpha = alpha
        self.intercept = False

    def __call__(self, x):
        return self.f_grad(x, return_gradient=False)


    def f_grad(self, x, return_gradient=True):
        if self.intercept:
            x_, c = x[:-1], x[-1]
        else:
            x_, c = x, 0.0
        z = safe_sparse_dot(self.A, x_, dense_output=True).ravel() + c
        loss = np.mean((1 - self.b) * z - self.logsig(z))
        penalty = safe_sparse_dot(x_.T, x_, dense_output=True).ravel()[0]
        loss += 0.5 * self.alpha * penalty

        if not return_gradient:
            return loss

        z0_b = self.expit_b(z, self.b)

        grad = safe_sparse_add(self.A.T.dot(z0_b) / self.A.shape[0], self.alpha * x_)
        grad = np.asarray(grad).ravel()
        grad_c = z0_b.mean()
        if self.intercept:
            return np.concatenate((grad, [grad_c]))

        return loss, grad

    def logsig(self, x):
        """Compute log(1 / (1 + exp(-t))) component-wise."""
        out = np.zeros_like(x)
        idx0 = x < -33
        out[idx0] = x[idx0]
        idx1 = (x >= -33) & (x < -18)
        out[idx1] = x[idx1] - np.exp(x[idx1])
        idx2 = (x >= -18) & (x < 37)
        out[idx2] = -np.log1p(np.exp(-x[idx2]))
        idx3 = x >= 37
        out[idx3] = -np.exp(-x[idx3])
        return out

    def expit_b(self, x, b):
        """Compute sigmoid(x) - b."""
        idx = x < 0
        out = np.zeros_like(x)
        exp_x = np.exp(x[idx])
        b_idx = b[idx]
        out[idx] = ((1 - b_idx) * exp_x - b_idx) / (1 + exp_x)
        exp_nx = np.exp(-x[~idx])
        b_nidx = b[~idx]
        out[~idx] = ((1 - b_nidx) - b_nidx * exp_nx) / (1 + exp_nx)
        return out

    @property
    def lipschitz(self):
        s = splinalg.svds(self.A, k=1, return_singular_vectors=False)[0]
        return 0.25 * (s * s) / self.A.shape[0] + self.alpha


def safe_sparse_add(a, b):
    if sparse.issparse(a) and sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # one of them is non-sparse, convert
        # everything to dense.
        if sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


class GroupL1:
    """
  Group Lasso penalty

  Parameters
  ----------

  alpha: float
      Constant multiplying this loss

  blocks: list of lists

  """

    def __init__(self, alpha, groups):
        self.alpha = alpha
        # groups need to be increasing
        for i, g in enumerate(groups):
            if not np.all(np.diff(g) == 1):
                raise ValueError("Groups must be contiguous")
            if i > 0 and groups[i - 1][-1] >= g[0]:
                raise ValueError("Groups must be increasing")
        self.groups = groups

    def __call__(self, x):
        return self.alpha * np.sum([np.linalg.norm(x[g]) for g in self.groups])

    def prox(self, x, step_size):
        out = x.copy()
        for g in self.groups:
            norm = np.linalg.norm(x[g])
            if isinstance(step_size,np.ndarray):
                out[g] = np.maximum(1-self.alpha/((1/step_size[g])*norm),0)*out[g]
            else:   
                if norm > self.alpha * step_size:
                    out[g] -= step_size * self.alpha * out[g] / norm
                else:
                    out[g] = 0
        return out


def loss(x,f,G1,G2):
    return f(x) + G1(x)+ G2(x)
