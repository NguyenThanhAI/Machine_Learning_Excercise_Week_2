import argparse
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from utils import construct_kernel


def pca(x: np.ndarray, alpha: float=0.95) -> Tuple[np.ndarray, np.ndarray]:
    
    mu = np.mean(x, axis=0, keepdims=True)

    x_mu = x - mu

    cov_matrix = np.matmul(x_mu.T, x_mu) / x.shape[0]

    w, v = np.linalg.eig(cov_matrix)

    order = np.argsort(w)[::-1]

    w = w[order]
    v = v[:, order]

    rate = np.cumsum(w) / np.sum(w)

    r = np.where(rate >= alpha)

    U = v[:, :(r[0][0] + 1)]

    reduced_x = np.matmul(x, U)

    return U, reduced_x



def kernel_pca(x: np.ndarray, alpha: float=0.95, type: str="gaussian_rbf", sigma: Optional[float]=None, r: Optional[float]=None, gamma: Optional[float]=None, d: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    if type == "linear":
        K = construct_kernel(x=x, type=type)
    elif type == "gaussian_rbf":
        K = construct_kernel(x=x, type=type, sigma=sigma)
    elif type == "polynomial":
        K = construct_kernel(x=x, type=type, r=r, gamma=gamma, d=d)
    elif type == "sigmoid":
        K = construct_kernel(x=x, type=type, r=r, gamma=gamma)
    else:
        raise ValueError("{} is not a supported kernel type".format(type))

    n = K.shape[0]

    assert K.shape[0] == K.shape[1]

    K = np.matmul(np.eye(n) - np.ones(shape=(n, n))/n, K)
    K = np.matmul(K, np.eye(n) - np.ones(shape=(n, n))/n)

    eta, c = np.linalg.eig(K)

    eta = np.real(eta)
    c = np.real(c)

    order = np.argsort(eta)[::-1]
    eta = eta[order]
    c = c[:, order]

    lamb_da = eta / n

    c = c / (np.sqrt(eta + 1e-8)[np.newaxis, :])

    rate = np.cumsum(lamb_da) / np.sum(lamb_da)

    r = np.where(rate >= alpha)

    C = c[:, :(r[0][0] + 1)]

    reduced_data = np.matmul(C.T, K).T

    return C, reduced_data


def svd(x: np.ndarray, alpha: float=0.98) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    w, v = np.linalg.eig(np.matmul(x.T, x))

    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]

    sigma = np.sqrt(w)

    rate = np.cumsum(w) / np.sum(w)

    q = np.where(rate >= alpha)

    v = v[:, :(q[0][0] + 1)]

    sigma = sigma[:(q[0][0] + 1)]

    u = np.matmul(x, v) / sigma[np.newaxis, :]

    return u, sigma, v



if __name__ == "__main__":

    