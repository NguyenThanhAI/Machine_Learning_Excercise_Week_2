from typing import Optional
import numpy as np


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



if __name__ == "__main__":

    np.random.seed(1000)

    x = np.random.uniform(low=0, high=101, size=(10, 3))

    for kernel_type in ("linear", "gaussian_rbf", "polynomial", "sigmoid"):
        if kernel_type == "linear":
            K = construct_kernel(x=x, type=kernel_type)
        elif kernel_type == "gaussian_rbf":
            K = construct_kernel(x=x, type=kernel_type, sigma=40)
        elif kernel_type == "polynomial":
            K = construct_kernel(x=x, type=kernel_type, r=0.1, gamma=0.0001, d=-1/3)
        elif kernel_type == "sigmoid":
            K = construct_kernel(x=x, type=kernel_type, r=0.1, gamma=0.0001)

        print("Kernel of type: {} is {}".format(kernel_type, K))
