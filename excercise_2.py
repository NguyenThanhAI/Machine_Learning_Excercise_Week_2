import argparse
import math
from typing import Tuple, Optional
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt


def construct_kernel(x: np.ndarray, type: str, sigma: Optional[float]=None, r: Optional[float]=None, gamma: Optional[float]=None, d: Optional[float]=None) -> np.ndarray:

    if type == "linear":
        return np.matmul(x, x.T)
    elif type == "gaussian_rbf":
        assert sigma is not None
        dist_matrix = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        square_dist_matrix = np.sum(dist_matrix**2, axis=2)
        return np.exp(-square_dist_matrix/(2*sigma**2))
    elif type == "polynomial":
        assert r is not None and gamma is not None and d is not None
        return (r + gamma * np.matmul(x, x.T))**d
    elif type == "sigmoid":
        assert r is not None and gamma is not None
        return np.tanh(r + gamma * np.matmul(x, x.T))
    else:
        raise ValueError("{} is not a supported kernel type".format(type))


def read_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    x = np.genfromtxt(data_path, delimiter=",", usecols=(0, 1, 2, 3))
    y = np.genfromtxt(data_path, delimiter=",", usecols=4, dtype=str)

    return x, y


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

    #print(reduced_x)

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
    #print(K)
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

    #print(rate)

    r = np.where(rate >= alpha)

    C = c[:, :(r[0][0] + 1)]

    reduced_data = np.matmul(C.T, K).T

    #print(reduced_data.shape)
    #print(reduced_data)

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

    data_path = "iris.data"
    names = ["setal_length", "setal_width", "petal_length", "petal_width"]
    color_dict = {'Iris-setosa': "red", 'Iris-versicolor': "green", 'Iris-virginica': "blue"}

    x, y = read_data(data_path=data_path)

   # Display data

    fig = plt.figure(figsize=(15, 10), facecolor='w')
    fig.suptitle("Iris Dataset", fontsize=14)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
    nums = len(list(combinations(range(x.shape[1]), r=2)))
    cols = 3
    rows = math.ceil(nums/cols)
    i = 0

    for couple in combinations(range(x.shape[1]), r=2):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title("{} - {}".format(names[couple[0]], names[couple[1]]))
        x_ax = x[:, couple[0]]
        y_ax = x[:, couple[1]]
        for g in np.unique(y):
            ix = np.where(y == g)
            ax.scatter(x_ax[ix], y_ax[ix], label=g, s=50)

        ax.set_xlabel(names[couple[0]])
        ax.set_ylabel(names[couple[1]])
        ax.set_aspect(abs((x_ax.max() - x_ax.min())/(y_ax.max() - y_ax.min()))*1.0)
        ax.grid()
        ax.legend()
        i += 1

    plt.show()

    for reduce_type in ("pca", "kernel_pca", "svd"):
        #print(reduce_type)
        if reduce_type == "pca":
            U, reduced_x = pca(x=x, alpha=0.95)
        elif reduce_type == "kernel_pca":
            C, reduced_x = kernel_pca(x=x, alpha=0.95, type="gaussian_rbf", sigma=6)
        elif reduce_type == "svd":
            u, sigma, v = svd(x=x, alpha=0.98)
            reduced_x =  np.matmul(u * sigma[np.newaxis, :], v.T)
            #print(reduced_x)
        
        fig = plt.figure(figsize=(20, 20), facecolor='w')
        fig.suptitle("{}".format(reduce_type.upper()), fontsize=14)
        #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.2)
        nums = len(list(combinations(range(reduced_x.shape[1]), r=2)))
        cols = 3
        rows = math.ceil(nums/cols)
        i = 0

        #fig, ax = plt.subplots(rows, cols)
        for couple in combinations(range(reduced_x.shape[1]), r=2):
            ax = fig.add_subplot(rows, cols, i + 1)
            ax.set_title("Axis {} - Axis {}".format(couple[0], couple[1]))
            x_ax = reduced_x[:, couple[0]]
            y_ax = reduced_x[:, couple[1]]
            for g in np.unique(y):
                ix = np.where(y == g)
                ax.scatter(x_ax[ix], y_ax[ix], label=g, s=50)

            ax.set_xlabel("Axis {}".format(couple[0]))
            ax.set_ylabel("Axis {}".format(couple[1]))
            ax.set_aspect(abs((x_ax.max() - x_ax.min())/(y_ax.max() - y_ax.min()))*1.0)
            ax.grid()
            ax.legend()
            i += 1

        plt.show()
