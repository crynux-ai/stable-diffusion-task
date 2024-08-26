from annotated_types import Gt, Ge, Le, Lt
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import Any, Union

from .controlnet_args import ControlnetArgs
from .scheduler_args import DPMSolverMultistep, EulerAncestralDiscrete, LCM
from ..types import FloatFractionAsInt, NonEmptyString
from ..version import VersionString


class RefinerArgs(BaseModel):
    model: NonEmptyString
    variant: str | None = "fp16"
    denoising_cutoff: FloatFractionAsInt = 80  # Not used if controlnet is enabled
    steps: Annotated[int, Gt(0), Le(100)] = 20


class LoraArgs(BaseModel):
    model: NonEmptyString
    weight_file_name: str = ""
    weight: FloatFractionAsInt = 100


class TaskConfig(BaseModel):
    image_width: int = 512
    image_height: int = 512
    steps: Annotated[int, Gt(0), Le(100)] = 1
    seed: Annotated[int, Ge(0), Lt(2147483648)] = 0
    num_images: Annotated[int, Gt(0), Le(10)] = 6
    safety_checker: bool = True
    cfg: Annotated[int, Ge(0), Le(20)] = 5


class BaseModelArgs(BaseModel):
    name: NonEmptyString
    variant: str | None = "fp16"


class InferenceTaskArgs(BaseModel):
    version: VersionString = VersionString.V1_0_0
    base_model: BaseModelArgs | NonEmptyString
    unet: str = ""
    prompt: NonEmptyString
    negative_prompt: str = ""
    task_config: TaskConfig
    lora: LoraArgs | None = None
    controlnet: ControlnetArgs | None = None
    scheduler: Union[
        DPMSolverMultistep,
        EulerAncestralDiscrete,
        LCM
    ] = Field(discriminator="method", default=DPMSolverMultistep())
    vae: str = ""
    refiner: RefinerArgs | None = None
    textual_inversion: str = ""

    def model_post_init(self, __context: Any) -> None:
        if isinstance(self.base_model, str):
            self.base_model = BaseModelArgs(name=self.base_model)
