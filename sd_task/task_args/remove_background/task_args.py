from typing import Literal, Optional

from PIL.Image import Image
from pydantic import BaseModel, Field, ConfigDict
from pydantic.json_schema import SkipJsonSchema


RemoveBackgroundModel = Literal[
    "isnet-general-use",
    "isnet-anime",
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
]


class RemoveBackgroundTaskArgs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image_dataurl: str = ""
    image: SkipJsonSchema[Optional[Image]] = Field(None, exclude=True, init_var=True)
    model: RemoveBackgroundModel = "u2net"
    return_mask: bool = False
    alpha_matting: bool = False
