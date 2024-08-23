from typing import Literal

from pydantic import BaseModel

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
    image_dataurl: str
    model: RemoveBackgroundModel = "u2net"
    return_mask: bool = False
    alpha_matting: bool = False
