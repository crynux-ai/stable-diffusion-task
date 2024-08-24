from typing import Literal, List

from pydantic import BaseModel


TaggerModel = Literal[
    "WD14 ViT v1",
    "WD14 ViT v2",
    "WD14 ConvNeXT v1",
    "WD14 ConvNeXT v2",
    "WD14 ConvNeXTV2 v1",
    "WD14 SwinV2 v1",
    "WD14 moat tagger v2",
    "ML-Danbooru Caformer dec-5-97527",
    "ML-Danbooru TResNet-D 6-30000",
]

class TaggerTaskArgs(BaseModel):
    image_dataurl: str
    model: TaggerModel = "WD14 moat tagger v2"
    threshold: float = 0.35
    tag_frac_threshold: float = 0.05
    count_threshold: int = 100
    additional_tags: List[str] = []
    search_tags: List[str] = []
    replace_tags: List[str] = []
    keep_tags: List[str] = []
    exclude_tags: List[str] = []