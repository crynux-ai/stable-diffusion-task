from annotated_types import Gt, Ge, Le, Lt
from pydantic import BaseModel
from typing_extensions import Annotated

from .controlnet_args import ControlnetArgs
from .types import FloatFractionAsInt, NonEmptyString


class RefinerArgs(BaseModel):
    model: NonEmptyString
    denoising_cutoff: FloatFractionAsInt = 80  # Not used if controlnet is enabled
    steps: Annotated[int, Gt(0), Le(100)] = 20


class LoraArgs(BaseModel):
    model: NonEmptyString
    weight: FloatFractionAsInt = 100


class TaskConfig(BaseModel):
    image_width: int = 512
    image_height: int = 512
    steps: Annotated[int, Gt(0), Le(100)] = 25
    seed: Annotated[int, Ge(0), Lt(2147483648)] = 0
    num_images: Annotated[int, Gt(0), Le(10)] = 6
    safety_checker: bool = True
    cfg: Annotated[int, Gt(0), Le(20)] = 5


class InferenceTaskArgs(BaseModel):
    base_model: NonEmptyString
    prompt: NonEmptyString
    negative_prompt: str = ""
    task_config: TaskConfig
    lora: LoraArgs | None = None
    controlnet: ControlnetArgs | None = None
    vae: str = ""
    refiner: RefinerArgs | None = None
    textual_inversion: str = ""
