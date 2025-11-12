import torch
import numpy as np
import os
import json
from typing import Tuple, NamedTuple, Any, Dict
from args import ModelParams, PipelineParams, OptimizationParams
from utils.geometry_utils import build_covariance_from_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH

class GaussianState(NamedTuple):
    active_sh_degree: int
    xyz: torch.Tensor
    features_dc: torch.Tensor
    features_ac: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    opacity: torch.Tensor
    max_radii2D: torch.Tensor
    xyz_gradient_accum: torch.Tensor
    denom: torch.Tensor

class GaussianModel:
    def __init__(self, sh_degree: int) -> None:
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_ac = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        
    def capture(self) -> Tuple[GaussianState, Dict[str, Any]]:
        gaussian_state = GaussianState(
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_ac,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
        )

        optimizer_state: Dict[str, Any] = self.optimizer.state_dict() if self.optimizer else {}
        return gaussian_state, optimizer_state
    
    def restore(self, captured_checkpoint: Tuple[GaussianState, Dict[str, Any]], training_args: OptimizationParams) -> None:
        gaussian_state, optimizer_state = captured_checkpoint
        self.active_sh_degree = gaussian_state.active_sh_degree
        self._xyz = gaussian_state.xyz
        self._features_dc = gaussian_state.features_dc
        self._features_ac = gaussian_state.features_ac
        self._scaling = gaussian_state.scaling
        self._rotation = gaussian_state.rotation
        self._opacity = gaussian_state.opacity
        self.max_radii2D = gaussian_state.max_radii2D
        self.xyz_gradient_accum = gaussian_state.xyz_gradient_accum
        self.denom = gaussian_state.denom

        if self.optimizer and optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)
    
    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_ac = self._features_ac
        return torch.cat((features_dc, features_ac), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_ac(self):
        return self._features_ac
        
    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self):
        return build_covariance_from_scaling_rotation(self._scaling, self._rotation)
    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcs(self, pcd: BasicPointCloud, cam_infos: int):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

    