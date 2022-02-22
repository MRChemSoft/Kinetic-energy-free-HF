import numpy as np
from vampyr import vampyr3d as vp

def calc_overlap(Phi_0, Phi_1):
    """Calculate the overlap matrix between the orbitals Phi_0 and Phi_1

    Parameters:
    Phi_0 : Orbital vector
    Phi_1 : Orbital vector

    Returns:
    Overlap matrix

    """

    m = np.empty((len(Phi_0), len(Phi_1)))
    for i in range(len(Phi_0)):
        for j in range(len(Phi_1)):
            m[i, j] = vp.dot(Phi_0[i], Phi_1[j])
    return m


def norm_vec(Phi):
    return np.array([phi.norm() for phi in Phi])
