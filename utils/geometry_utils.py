import torch
from jaxtyping import Float, Int, Tuple
from torch import Tensor

def strip_symmetric(L: Float[Tensor, "N 3 3"]) -> Float[Tensor, "N 6"]:
    upper_coeffs = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    upper_coeffs[:,0] = L[:,0,0]
    upper_coeffs[:,1] = L[:,0,1]
    upper_coeffs[:,2] = L[:,0,2]
    upper_coeffs[:,3] = L[:,1,1]
    upper_coeffs[:,4] = L[:,1,2]
    upper_coeffs[:,5] = L[:,2,2]

    return upper_coeffs

def build_rotation(r: Float[Tensor, "N 4"]) -> Float[Tensor, "N 3 3"]:
    norm = torch.sqrt(r[:,0]*r[:,0] +
                      r[:,1]*r[:,1] +
                      r[:,2]*r[:,2] +
                      r[:,3]*r[:,3])
    
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    w = q[:,0]
    x = q[:,1]
    y = q[:,2]
    z = q[:,3]

    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - w*z)
    R[:,0,2] = 2*(x*z + w*y)
    R[:,1,0] = 2*(x*y + w*z)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - w*x)
    R[:,2,0] = 2*(x*z - w*y)
    R[:,2,1] = 2*(y*z + w*x)
    R[:,2,2] = 1 - 2*(x*x + y*y)

    return R

def build_scaling_rotation(s: Float[Tensor, "N 3"], r: Float[Tensor, "N 4"]) -> Float[Tensor, "N 3 3"]:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def build_covariance_from_scaling_rotation(scaling: Float[Tensor, "N 3"], rotation: Float[Tensor, "N 4"]) -> Float[Tensor, "N 6"]:
    L = build_scaling_rotation(scaling, rotation)
    actual_covariance = L @ L.transpose(1,2)
    symmetric_part = strip_symmetric(actual_covariance)
    return symmetric_part