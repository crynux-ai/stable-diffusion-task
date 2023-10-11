from pydantic import BaseModel


class CannyArgs(BaseModel):
    low_threshold: int
    high_threshold: int


class PreprocessArgs(BaseModel):
    method: str
    args: CannyArgs | None = None


class ControlnetArgs(BaseModel):
    model: str
    image_dataurl: str
    weight: float = 0.5
    preprocess: PreprocessArgs | None
