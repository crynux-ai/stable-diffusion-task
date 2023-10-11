from pydantic import BaseModel
from controlnet_args import ControlnetArgs


class RefinerArgs(BaseModel):
    model: str
    denoising_cutoff: float = 0.8    # Not used if controlnet is enabled
    steps: int = 20


class LoraArgs(BaseModel):
    model: str
    weight: float = 1.0


class TaskConfig(BaseModel):
    image_width: int = 512
    image_height: int = 512
    steps: int = 25
    seed: int = 0
    num_images: int = 6
    safety_checker: bool = True
    cfg: float = 5.0


class GenImageArgs(BaseModel):
    base_model: str
    prompt: str
    negative_prompt: str = ""
    task_config: TaskConfig
    lora: LoraArgs | None = None
    controlnet: ControlnetArgs | None = None
    vae: str = ""
    refiner: RefinerArgs | None = None
