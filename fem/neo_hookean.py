import warp as wp
import numpy as np
from .params import *

@wp.func
def tangent_stiffness(F: wp.mat33, dF: wp.mat33) -> wp.mat33:
    '''
    neo-hookean model
    '''
    F_inv_T = wp.transpose(wp.inverse(F))
    B = wp.inverse(F) @ dF
    det_F = wp.determinant(F)
    
    return mu * dF + (mu - lam * wp.log(det_F)) * F_inv_T @ wp.transpose(dF) @ F_inv_T + (lam * wp.trace(B)) * F_inv_T 

@wp.func
def PK1(F: wp.mat33) -> wp.mat33:
    '''
    neo-hookean
    '''
    F_inv_T = wp.transpose(wp.inverse(F))
    J = wp.determinant(F)
    return mu * (F - F_inv_T) + lam * wp.log(J) * F_inv_T

@wp.func
def psi(F: wp.mat33) -> float:
    I1 = wp.trace(wp.transpose(F) @ F)
    J = wp.determinant(F)
    logJ = wp.log(J)
    return mu * 0.5 * (I1 -3.0) - mu * logJ + lam * 0.5 * logJ * logJ
    