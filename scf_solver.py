import numpy as np
import numpy.linalg as LA
from  vampyr import vampyr3d as vp
from Operators import CouloumbOperator, ExchangeOperator, NuclearOperator, HelmholtzOperator

def scf_solver(Phi_n, fock_n, atoms, mra, epsilon):
    """Kinetric free Hartree-Fock SCF solver

    Paramters:
    Phi_n : Starting guess orbitals
    fock_n : Starting guess for Fock Matrix
    atoms : list of list contating coordinates and charge or atoms
    mra : The Multiresolution analysis to work on
    epsilon : Precision requirement

    Returns:
    Updates : Norm difference of each orbitals at each iteration
    fock_matrices : List of the fock matrices for each iteration
    Phi_n : Converged orbital vector

    """

    thrs = epsilon*10

    update = np.ones(len(Phi_n))
    updates = []
    energies = []

    V_nuc = NuclearOperator(mra, atoms, prec=epsilon)

    while (max(update) > thrs):

        J_n = CouloumbOperator(Phi_n, mra, prec=epsilon)
        K_n = ExchangeOperator(Phi_n, mra, prec=epsilon)
        H = HelmholtzOperator(epsilon=np.diag(fock_n), mra=mra, prec=epsilon)

        Phi_np1 = -2.0*H(V_nuc(Phi_n) + J_n(Phi_n) - K_n(Phi_n))

        # Update fock matrix
        d_Phi_n = Phi_np1 - Phi_n
        update = np.array([d_phi_n.norm() for d_phi_n in d_Phi_n])
        s1 = calc_overlap(d_Phi_n, Phi_n)
        s2 = calc_overlap(Phi_np1, d_Phi_n)

        eig, U = LA.eig(calc_overlap(Phi_np1, Phi_np1))
        Sm5 = U @ np.diag(eig**(-0.5)) @ U.transpose()

        Phi_np1_norm = Sm5 @ Phi_np1

        J_np1 = CouloumbOperator(Phi_np1_norm, mra, prec=epsilon)
        K_np1 = ExchangeOperator(Phi_np1_norm, mra, prec=epsilon)
        V_dPhi_n = V_nuc(d_Phi_n) + J_n(d_Phi_n) - K_n(d_Phi_n)
        o3 = calc_overlap(Phi_np1, V_dPhi_n)
        dV_Phi_np1 = J_np1(Phi_np1) - K_np1(Phi_np1) -(J_n(Phi_np1) - K_n(Phi_np1))
        o4 = calc_overlap(Phi_np1, dV_Phi_np1)


        fock_n = fock_n + s1 @ fock_n + s2 @ fock_n + o3 + o4

        # Diagonalize fock matrix
        fock_n = Sm5.transpose() @ fock_n @ Sm5
        U = LA.eig(fock_n)[1]
        fock_n = np.diag(LA.eig(fock_n)[0])
        Phi_n = U.transpose() @ Phi_np1_norm

        updates.append(update)
        energies.append(calc_energies(atoms, mra, epsilon, fock_n, Phi_n, V_nuc))


    return np.array(updates), energies, Phi_n


def calc_energies(atoms, mra, epsilon, fock_n, Phi_n, V_nuc):
    """"Calcuate all energy contributions"""

    J = CouloumbOperator(Phi_n, mra, prec=epsilon)
    K = ExchangeOperator(Phi_n, mra, prec=epsilon)


    sum_orb_energy = 2.0*np.trace(fock_n)
    couloumb_energy = np.trace(calc_overlap(Phi_n, J(Phi_n)))
    exchange_energy = -np.trace(calc_overlap(Phi_n, K(Phi_n)))
    electron_nuclear_energy = 2*np.trace(calc_overlap(Phi_n, V_nuc(Phi_n)))
    total_energy = sum_orb_energy - couloumb_energy - exchange_energy
    kinetic_energy = total_energy - couloumb_energy - exchange_energy - electron_nuclear_energy
    return {"$\sum_i \epsilon_i$": sum_orb_energy, "$E_{coul}$":couloumb_energy,
                    "$E_{en}$":electron_nuclear_energy,
                    "$E_{ex}$": exchange_energy, "$E_{kin}$":kinetic_energy,
                    "$E_{tot}$":total_energy}


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

def starting_guess(mra, prec, nr_of_orbitals, center):
    """Primitive starting guess, works for Be"""

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
