# Analytic guess for solution
from non_kintetic import calc_overlap
from vampyr import vampyr3d as vp
import numpy as np

import numpy.linalg as LA
import numpy as np



def starting_guess(mra, prec, nr_of_orbitals, center):

    P_eps = vp.ScalingProjector(mra=mra, prec=prec)
    Phi = []
    for i in range(1, nr_of_orbitals+1):
        def gauss(r):
            R = np.sqrt((r[0]-center[0])**2 + (r[1]-center[1])**2 + (r[2] - center[2])**2)
            return np.exp(-R*R/i)

        orb = P_eps(gauss)
        orb.normalize()
        Phi.append(orb)
    Phi = np.array(Phi)

    eig, U = LA.eig(calc_overlap(Phi, Phi))
    Sm5 = U @ np.diag(eig**(-0.5)) @ U.transpose()
    return Sm5 @ Phi
