from typing import Literal
from pydantic import BaseModel


class CommonSchedulerArgs(BaseModel):
    num_train_timesteps: int | None = None
    beta_start: float | None = None
    beta_end: float | None = None
    beta_schedule: str | None = None
    prediction_type: str | None = None
    timestep_spacing: str | None = None
    steps_offset: int | None = None
    rescale_betas_zero_snr: bool | None = None


class EulerAncestralDiscrete(BaseModel):
    method: Literal['EulerAncestralDiscreteScheduler'] = 'EulerAncestralDiscreteScheduler'
    args: CommonSchedulerArgs | None = None


class LCMArgs(CommonSchedulerArgs):
    original_inference_steps: int | None = None
    clip_samples: int | None = None
    clip_samples_range: float | None = None
    set_alpha_to_one: bool | None = None
    thresholding: bool | None = None
    dynamic_thresholding_ratio: float | None = None
    sample_max_value: float | None = None
    timestep_scaling: float | None = None


class LCM(BaseModel):
    method: Literal['LCMScheduler'] = 'LCMScheduler'
    args: LCMArgs | None = None


class DPMSolverMultistepArgs(CommonSchedulerArgs):
    solver_order: int | None = None
    thresholding: bool | None = None
    dynamic_thresholding_ratio: float | None = None
    sample_max_value: float | None = None
    algorithm_type: str | None = None
    solver_type: str | None = None
    lower_order_final: bool | None = None
    euler_at_final: bool | None = None
    use_karras_sigmas: bool | None = None
    use_lu_lambdas: bool | None = None
    final_sigmas_type: str | None = None
    lambda_min_clipped: float | None = None
    variance_type: str | None = None


class DPMSolverMultistep(BaseModel):
    method: Literal['DPMSolverMultistepScheduler'] = 'DPMSolverMultistepScheduler'
    args: DPMSolverMultistepArgs | None = None
