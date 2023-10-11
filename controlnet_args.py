from pydantic import BaseModel, Field
from typing import Literal, Union


class CannyArgs(BaseModel):
    low_threshold: int = 100
    high_threshold: int = 200


class PreprocessMethodCanny(BaseModel):
    method: str = Literal['canny']
    args: CannyArgs | None = None


class PreprocessMethodScribbleHED(BaseModel):
    method: str = Literal['scribble_hed']


class PreprocessMethodSoftEdgeHED(BaseModel):
    method: str = Literal['softedge_hed']


class PreprocessMethodScribbleHEDSafe(BaseModel):
    method: str = Literal['scribble_hedsafe']


class PreprocessMethodSoftEdgeHEDSafe(BaseModel):
    method: str = Literal['softedge_hedsafe']


class PreprocessMethodDepthMidas(BaseModel):
    method: str = Literal['depth_midas']


class PreprocessMethodMLSD(BaseModel):
    method: str = Literal['mlsd']


class PreprocessMethodOpenPose(BaseModel):
    method: str = Literal['openpose']


class ControlnetArgs(BaseModel):
    model: str
    image_dataurl: str
    weight: float = 0.5
    preprocess: Union[
        PreprocessMethodCanny,
        PreprocessMethodScribbleHED,
        PreprocessMethodScribbleHEDSafe,
        PreprocessMethodSoftEdgeHEDSafe,
        PreprocessMethodDepthMidas,
        PreprocessMethodMLSD,
        PreprocessMethodOpenPose
    ] = Field(discriminator="method", default=None)
