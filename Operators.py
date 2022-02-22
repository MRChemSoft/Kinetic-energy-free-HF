import numpy as np
from vampyr import vampyr3d as vp

class ExchangeOperator():
    """
    Vectorized Exchange operator

    Parameters
    ----------

    Phi : Orbital vector
    mra : The multiresolution analysis we work on
    prec : Precision requirement

    Attributes
    ----------
    Phi : Orbitals
    mra : mra
    prec : Precision
    P : Poisson Operator

    """
    def __init__(self, Psi, mra, prec=1.0e-4):
        self.Psi = Psi
        self.mra = mra
        self.prec = prec
        self.P = vp.PoissonOperator(mra=mra, prec=self.prec)

    def __call__(self, Phi):
        """Operate the excahnge operator onto an orbital Vector Phi"""

        Phi_out = []
        for j in range(len(Phi)):
            tmp = (self.Psi[0]*self.P(Phi[j]*self.Psi[0])).crop(self.prec)
            for i in range(1, len(self.Psi)):
                tmp += (self.Psi[i]*self.P((Phi[j]*self.Psi[i]).crop(self.prec))).crop(self.prec)
            Phi_out.append(tmp)
        return 4*np.pi*np.array(Phi_out)


class CouloumbOperator():
    """
    Vectorized Couloumb operator

    Parameters
    ----------

    Phi : Orbital vector
    mra : The multiresolution analysis we work on
    prec : Precision requirement

    Attributes
    ----------
    Phi : Orbitals
    mra : mra
    prec : Precision
    P : Poisson Operator
    rho : Electronic density

    """
    def __init__(self, Phi, mra, prec=1.0e-4):
        self.Phi = Phi
        self.mra = mra
        self.prec = prec
        self.P = vp.PoissonOperator(mra=mra, prec=self.prec)
        self.rho = None
        self.setup()

    def setup(self):
        tmp = (self.Phi[0]**2).crop(self.prec)
        for i in range(1, len(self.Phi)):
            tmp += (self.Phi[i]**2).crop(self.prec)
        self.rho = (2.0*4.0*np.pi)*self.P(tmp).crop(self.prec)

    def __call__(self, Phi):
        """Operate Couloumb operator onto an orbital vector Phi"""
        return self.rho*Phi


class NuclearOperator():
    """
    Vectorized Nuclear potential operator

    Parameters
    ----------

    mra : The multiresolution analysis we work on
    atoms : Atoms, list of list containing charge and coordinates of the atoms
    prec : Precision requirement

    Attributes
    ----------
    mra : mra
    atoms: atoms
    rho : Nuclear potential

    """
    def __init__(self, mra, atoms, prec=1.0e-4):
        self.mra = mra
        self.prec= prec
        self.atoms = atoms
        self.rho = None
        self.setup()

    def setup(self):
        adaptive_projector = vp.ScalingProjector(mra=self.mra, prec=self.prec)
        f = NuclearFunction(self.atoms)
        self.rho = adaptive_projector(f)

    def __call__(self, Phi):
        """Operate Nuclear potential operator onto an orbital vector"""
        return Phi*self.rho.crop(self.prec)

class NuclearFunction():
    """
    Vectorized Nuclear potential operator

    Parameters
    ----------

    atoms : Atoms, list of list containing charge and coordinates of the atoms

    Attributes
    ----------
    atoms: atoms

    """


    def __init__(self, atoms):
        self.atoms = atoms

    def __call__(self, R):
        "Returns the nuclear potential value in R"
        tmp = 0
        for atom in self.atoms:
            Z = atom[0]
            R0 = atom[1]
            tmp += -Z / ((R0[0] - R[0])**2 + (R0[1] - R[1])**2 + (R0[2] - R[2])**2)**0.5
        return tmp

class HelmholtzOperator():
    """
    Vectorized helmholtz operator

    Parameters
    ----------
    epsilon : vector of orbital energies
    mra : The multiresolution analysis
    prec : Precision

    Attributes
    ----------
    epsilon : orbital energies
    mra : mra
    prec : precision
    operators : list containing HelmholtzOperators for each orbital

    """
    def __init__(self, epsilon, mra, prec=1.0e-4):
        self.epsilon = epsilon
        self.mra = mra
        self.prec = prec
        self.operators = []
        self.setup()

    def setup(self):
        mu = [np.sqrt(-2.0*e) if e < 0 else np.sqrt(2.0) for e in self.epsilon]
        for m in mu:
            self.operators.append(vp.HelmholtzOperator(mra=self.mra, exp=m, prec=self.prec))

    def __call__(self, potential):
        """Operate the Helmholtz operator onto an orbital vector"""
        return np.array([self.operators[i](potential[i]) for i in range(len(potential))])
