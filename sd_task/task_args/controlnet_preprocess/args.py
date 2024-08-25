from typing import Literal, Optional, Union

from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from ..types import FloatFractionAsInt


class CannyArgs(BaseModel):
    low_threshold: int = 100
    high_threshold: int = 200


class PreprocessMethodCanny(BaseModel):
    method: Literal["canny"] = "canny"
    args: CannyArgs | None = None


class PreprocessMethodScribbleHED(BaseModel):
    method: Literal["scribble_hed"] = "scribble_hed"


class PreprocessMethodSoftEdgeHED(BaseModel):
    method: Literal["softedge_hed"] = "softedge_hed"


class PreprocessMethodScribbleHEDSafe(BaseModel):
    method: Literal["scribble_hedsafe"] = "scribble_hedsafe"


class PreprocessMethodSoftEdgeHEDSafe(BaseModel):
    method: Literal["softedge_hedsafe"] = "softedge_hedsafe"


class PreprocessMethodDepthMidas(BaseModel):
    method: Literal["depth_midas"] = "depth_midas"


class MLSDArgs(BaseModel):
    thr_v: FloatFractionAsInt = 10
    thr_d: FloatFractionAsInt = 10


class PreprocessMethodMLSD(BaseModel):
    method: Literal["mlsd"] = "mlsd"
    args: MLSDArgs | None = None


class PreprocessMethodOpenPoseBodyOnly(BaseModel):
    method: Literal["openpose"] = "openpose"


class PreprocessMethodOpenPoseFaceAndBody(BaseModel):
    method: Literal["openpose_face"] = "openpose_face"


class PreprocessMethodOpenPoseFaceOnly(BaseModel):
    method: Literal["openpose_faceonly"] = "openpose_faceonly"


class PreprocessMethodOpenPoseFull(BaseModel):
    method: Literal["openpose_full"] = "openpose_full"


class PreprocessMethodOpenPoseHand(BaseModel):
    method: Literal["openpose_hand"] = "openpose_hand"


class PreprocessMethodDWPose(BaseModel):
    method: Literal["dwpose"] = "dwpose"


class PidiNetArgs(BaseModel):
    apply_filter: bool = False


class PreprocessMethodScribblePidiNet(BaseModel):
    method: Literal["scribble_pidinet"] = "scribble_pidinet"
    args: PidiNetArgs | None = None


class PreprocessMethodSoftEdgePidiNet(BaseModel):
    method: Literal["softedge_pidinet"] = "softedge_pidinet"
    args: PidiNetArgs | None = None


class PreprocessMethodScribblePidiNetSafe(BaseModel):
    method: Literal["scribble_pidisafe"] = "scribble_pidisafe"
    args: PidiNetArgs | None = None


class PreprocessMethodSoftEdgePidiNetSafe(BaseModel):
    method: Literal["softedge_pidisafe"] = "softedge_pidisafe"
    args: PidiNetArgs | None = None


class PreprocessMethodNormalBAE(BaseModel):
    method: Literal["normal_bae"] = "normal_bae"


class PreprocessMethodLineartCoarse(BaseModel):
    method: Literal["lineart_coarse"] = "lineart_coarse"


class PreprocessMethodLineartRealistic(BaseModel):
    method: Literal["lineart_realistic"] = "lineart_realistic"


class PreprocessMethodLineartAnime(BaseModel):
    method: Literal["lineart_anime"] = "lineart_anime"


class DepthZoeArgs(BaseModel):
    gamma_corrected: bool = False


class PreprocessMethodDepthZoe(BaseModel):
    method: Literal["depth_zoe"] = "depth_zoe"
    args: DepthZoeArgs | None = None


class LeresArgs(BaseModel):
    thr_a: int = 0
    thr_b: int = 0


class PreprocessMethodDepthLeres(BaseModel):
    method: Literal["depth_leres"] = "depth_leres"
    args: LeresArgs | None = None


class PreprocessMethodDepthLeresPP(BaseModel):
    method: Literal["depth_leres++"] = "depth_leres++"
    args: LeresArgs | None = None


class ShuffleArgs(BaseModel):
    h: int | None = None
    w: int | None = None
    f: int | None = None


class PreprocessMethodShuffle(BaseModel):
    method: Literal["shuffle"] = "shuffle"
    args: ShuffleArgs | None = None


class MediapipeFaceArgs(BaseModel):
    max_faces: int = 1
    min_confidence: FloatFractionAsInt = 50


class PreprocessMethodMediapipeFace(BaseModel):
    method: Literal["mediapipe_face"] = "mediapipe_face"
    args: MediapipeFaceArgs | None = None


class ControlnetPreprocessTaskArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_dataurl: str = ""
    image: SkipJsonSchema[Optional[Image]] = Field(None, exclude=True, init_var=True)
    preprocess: Union[
        PreprocessMethodCanny,
        PreprocessMethodScribbleHED,
        PreprocessMethodScribbleHEDSafe,
        PreprocessMethodSoftEdgeHEDSafe,
        PreprocessMethodDepthMidas,
        PreprocessMethodMLSD,
        PreprocessMethodOpenPoseBodyOnly,
        PreprocessMethodOpenPoseFaceAndBody,
        PreprocessMethodOpenPoseFaceOnly,
        PreprocessMethodOpenPoseFull,
        PreprocessMethodOpenPoseHand,
        PreprocessMethodScribblePidiNet,
        PreprocessMethodSoftEdgePidiNet,
        PreprocessMethodScribblePidiNetSafe,
        PreprocessMethodSoftEdgePidiNetSafe,
        PreprocessMethodNormalBAE,
        PreprocessMethodLineartCoarse,
        PreprocessMethodLineartRealistic,
        PreprocessMethodLineartAnime,
        PreprocessMethodDepthZoe,
        PreprocessMethodDepthLeres,
        PreprocessMethodDepthLeresPP,
        PreprocessMethodShuffle,
        PreprocessMethodMediapipeFace,
    ]
