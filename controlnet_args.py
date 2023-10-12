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


class MLSDArgs(BaseModel):
    thr_v: float = 0.1
    thr_d: float = 0.1


class PreprocessMethodMLSD(BaseModel):
    method: str = Literal['mlsd']
    args: MLSDArgs | None = None


class PreprocessMethodOpenPoseBodyOnly(BaseModel):
    method: str = Literal['openpose']


class PreprocessMethodOpenPoseFaceAndBody(BaseModel):
    method: str = Literal['openpose_face']


class PreprocessMethodOpenPoseFaceOnly(BaseModel):
    method: str = Literal['openpose_faceonly']


class PreprocessMethodOpenPoseFull(BaseModel):
    method: str = Literal['openpose_full']


class PreprocessMethodOpenPoseHand(BaseModel):
    method: str = Literal['openpose_hand']


class PreprocessMethodDWPose(BaseModel):
    method: str = Literal['dwpose']


class PidiNetArgs(BaseModel):
    apply_filter: bool = False


class PreprocessMethodScribblePidiNet(BaseModel):
    method: str = Literal['scribble_pidinet']
    args: PidiNetArgs | None = None


class PreprocessMethodSoftEdgePidiNet(BaseModel):
    method: str = Literal['softedge_pidinet']
    args: PidiNetArgs | None = None


class PreprocessMethodScribblePidiNetSafe(BaseModel):
    method: str = Literal['scribble_pidisafe']
    args: PidiNetArgs | None = None


class PreprocessMethodSoftEdgePidiNetSafe(BaseModel):
    method: str = Literal['softedge_pidisafe']
    args: PidiNetArgs | None = None


class PreprocessMethodNormalBAE(BaseModel):
    method: str = Literal['normal_bae']


class PreprocessMethodLineartCoarse(BaseModel):
    method: str = Literal['lineart_coarse']


class PreprocessMethodLineartRealistic(BaseModel):
    method: str = Literal['lineart_realistic']


class PreprocessMethodLineartAnime(BaseModel):
    method: str = Literal['lineart_anime']


class DepthZoeArgs(BaseModel):
    gamma_corrected: bool = False


class PreprocessMethodDepthZoe(BaseModel):
    method: str = Literal['depth_zoe']
    args: DepthZoeArgs | None = None


class LeresArgs(BaseModel):
    thr_a: int = 0
    thr_b: int = 0


class PreprocessMethodDepthLeres(BaseModel):
    method: str = Literal['depth_leres']
    args: LeresArgs | None = None


class PreprocessMethodDepthLeresPP(BaseModel):
    method: str = Literal['depth_leres++']
    args: LeresArgs | None = None


class ShuffleArgs(BaseModel):
    h: int | None = None
    w: int | None = None
    f: int | None = None


class PreprocessMethodShuffle(BaseModel):
    method: str = Literal['shuffle']
    args: ShuffleArgs | None = None


class MediapipeFaceArgs(BaseModel):
    max_faces: int = 1
    min_confidence: float = 0.5


class PreprocessMethodMediapipeFace(BaseModel):
    method: str = Literal['mediapipe_face']
    args: MediapipeFaceArgs | None = None


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
        PreprocessMethodMediapipeFace
    ] = Field(discriminator="method", default=None)
