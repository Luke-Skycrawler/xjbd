import warp as wp
import numpy as np
# from .params import *

@wp.func
def tangent_stiffness(F: wp.mat33, dF: wp.mat33, lam: float, mu: float) -> wp.mat33:
    '''
    dP = mu(dF + dF^T) + lam tr(dF) I
    '''
    return mu * (dF + wp.transpose(dF)) + lam * wp.trace(dF) * wp.identity(3, dtype = float)

@wp.func
def psi(F: wp.mat33, lam: float, mu: float) -> float:
    '''
    psi = mu eps : eps + lam/2 (tr(eps))^2
    '''
    eps = 0.5 * (F + wp.transpose(F)) - wp.identity(3, dtype = float)
    # norm_eps = wp.trace(wp.transpose(eps) @ eps)
    norm_eps = wp.ddot(eps, eps)
    tre = wp.trace(eps)
    return mu * norm_eps + lam * 0.5 * tre * tre

@wp.func
def PK1(F: wp.mat33, lam: float, mu: float) -> wp.mat33:
    '''
    P = 2 mu eps + lam tr(eps) I
    '''
    eps = 0.5 * (F + wp.transpose(F)) - wp.identity(3, dtype = float)
    return 2.0 * mu * eps + lam * wp.trace(eps) * wp.identity(3, dtype = float)
    