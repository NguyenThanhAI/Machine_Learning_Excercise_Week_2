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
        