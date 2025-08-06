import warp as wp
# from .params import *

@wp.func
def tangent_stiffness(F: wp.mat33, dF: wp.mat33, lam: float, mu: float) -> wp.mat33:
    '''
    dP = dF (2 mu E + lam tr(E) I) + F (2 mu dE + lam tr(dE) I)
    '''
    E = 0.5 * (wp.transpose(F) @ F - wp.identity(3, dtype = float))
    dE = wp.transpose(dF) @ F
    dE = 0.5 * (dE + wp.transpose(dE))
    return dF @ (2.0 * mu * E + lam * wp.trace(E) * wp.identity(3, dtype = float)) + F @ (2.0 * mu * dE + lam * wp.trace(dE) * wp.identity(3, dtype = float))

@wp.func
def psi(F: wp.mat33, lam: float, mu: float) -> float:
    '''
    E = (F^T F - I) / 2
    psi = mu E : E + lam/2 (tr(E))^2
    '''
    E = 0.5 * (wp.transpose(F) @ F - wp.identity(3, dtype = float))
    # norm_E = wp.trace(wp.transpose(E) @ E)
    norm_E = wp.ddot(E, E)
    trE = wp.trace(E)
    return mu * norm_E + lam * 0.5 * trE * trE

@wp.func
def PK1(F: wp.mat33, lam: float, mu: float) -> wp.mat33:
    '''
    P = F (2 mu E + lam tr(E) I)
    '''

    E = 0.5 * (wp.transpose(F) @ F - wp.identity(3, dtype = float))
    return F @ (2.0 * mu * E + lam * wp.trace(E) * wp.identity(3, dtype = float))
    