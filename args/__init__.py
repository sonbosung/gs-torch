import typer
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
import os

@dataclass
class ModelParams:
    sh_degree: int = 3
    source_path: str = ""
    model_path: str = ""
    images: str = "images"
    depths: str = ""
    resolution: int = -1
    white_background: bool = False
    train_test_exp: bool = False
    data_device: str = "cuda"
    eval: bool = False

    def __post_init__(self):
        if self.source_path:
            self.source_path = os.path.abspath(self.source_path)

@dataclass
class PipelineParams:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False

@dataclass
class OptimizationParams:
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    lambda_dssim: float = 0.2
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.0002
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.01
    random_background: bool = False

def create_model_params(
    sh_degree: int = typer.Option(3, help="Spherical harmonics degree"),
    source_path: str = typer.Option("", "-s", "--source_path", help="Path to source data"),
    model_path: str = typer.Option("", "-m", "--model_path", help="Path to save model"),
    images: str = typer.Option("images", "-i", "--images", help="Images folder name"),
    depths: str = typer.Option("", help="Depths folder name"),
    resolution: int = typer.Option(-1, "-r", "--resolution", help="Resolution for training"),
    white_background: bool = typer.Option(False, help="Use white background"),
    train_test_exp: bool = typer.Option(False, help="Train test exposure"),
    data_device: str = typer.Option("cuda", help="Device for data loading"),
    eval: bool = typer.Option(False, help="Evaluation mode"),
) -> ModelParams:
    return ModelParams(
        sh_degree=sh_degree,
        source_path=source_path,
        model_path=model_path,
        images=images,
        depths=depths,
        resolution=resolution,
        white_background=white_background,
        train_test_exp=train_test_exp,
        data_device=data_device,
        eval=eval,
    )

def create_pipeline_params(
    convert_SHs_python: bool = typer.Option(False, help="Convert SHs in Python"),
    compute_cov3D_python: bool = typer.Option(False, help="Compute 3D covariance in Python"),
    debug: bool = typer.Option(False, help="Enable debug mode"),
    antialiasing: bool = typer.Option(False, help="Enable antialiasing"),
) -> PipelineParams:
    """Create PipelineParams from command line arguments"""
    return PipelineParams(
        convert_SHs_python=convert_SHs_python,
        compute_cov3D_python=compute_cov3D_python,
        debug=debug,
        antialiasing=antialiasing,
    )

def create_optimization_params(
    iterations: int = typer.Option(30_000, help="Number of training iterations"),
    position_lr_init: float = typer.Option(0.00016, help="Initial learning rate for positions"),
    position_lr_final: float = typer.Option(0.0000016, help="Final learning rate for positions"),
    position_lr_delay_mult: float = typer.Option(0.01, help="Position LR delay multiplier"),
    position_lr_max_steps: int = typer.Option(30_000, help="Max steps for position LR"),
    feature_lr: float = typer.Option(0.0025, help="Learning rate for features"),
    opacity_lr: float = typer.Option(0.025, help="Learning rate for opacity"),
    scaling_lr: float = typer.Option(0.005, help="Learning rate for scaling"),
    rotation_lr: float = typer.Option(0.001, help="Learning rate for rotation"),
    lambda_dssim: float = typer.Option(0.2, help="DSSIM loss weight"),
    densification_interval: int = typer.Option(100, help="Densification interval"),
    opacity_reset_interval: int = typer.Option(3000, help="Opacity reset interval"),
    densify_from_iter: int = typer.Option(500, help="Start densification from this iteration"),
    densify_until_iter: int = typer.Option(15_000, help="Stop densification at this iteration"),
    densify_grad_threshold: float = typer.Option(0.0002, help="Gradient threshold for densification"),
    depth_l1_weight_init: float = typer.Option(1.0, help="Initial depth L1 weight"),
    depth_l1_weight_final: float = typer.Option(0.01, help="Final depth L1 weight"),
    random_background: bool = typer.Option(False, help="Use random background"),
) -> OptimizationParams:
    """Create OptimizationParams from command line arguments"""
    return OptimizationParams(
        iterations=iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
        lambda_dssim=lambda_dssim,
        densification_interval=densification_interval,
        opacity_reset_interval=opacity_reset_interval,
        densify_from_iter=densify_from_iter,
        densify_until_iter=densify_until_iter,
        densify_grad_threshold=densify_grad_threshold,
        depth_l1_weight_init=depth_l1_weight_init,
        depth_l1_weight_final=depth_l1_weight_final,
        random_background=random_background,
    )