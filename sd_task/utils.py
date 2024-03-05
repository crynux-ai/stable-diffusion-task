import enum
import platform
from typing import Union

class Platform(enum.Enum):
    UNKNONW = 0
    LINUX_CUDA = 1
    MACOS_MPS = 2

def get_platform() -> Union[Platform, str]:
    res = platform.system()
    if res == "Linux":
        return Platform.LINUX_CUDA
    elif res == "Darwin":
        return Platform.MACOS_MPS
    else:
        return res

def get_accelerator():
    platform = get_platform()    
    if platform == Platform.LINUX_CUDA:
        return "cuda"
    elif platform == Platform.MACOS_MPS:
        return "mps"
    else:
        return "cpu"
