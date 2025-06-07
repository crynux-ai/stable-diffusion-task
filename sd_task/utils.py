import base64
import platform
import re
from io import BytesIO

from PIL import Image


def get_accelerator():
    """Detect available accelerator type"""
    if platform.system() == "Darwin":
        try:
            import torch.mps

            return "mps"
        except ImportError:
            pass

    try:
        import torch.cuda

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


def get_gpu_info():
    """Get detailed GPU information"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            return {
                "gpu_count": gpu_count,
                "current_device": current_device,
                "gpu_name": gpu_name,
                "gpu_memory_gb": round(gpu_memory_gb, 2),
                "compute_capability": torch.cuda.get_device_capability(current_device)
            }
    except ImportError:
        pass
    
    return None


def is_rtx_5090():
    """Detect if the graphics card is RTX 5090"""
    gpu_info = get_gpu_info()
    if gpu_info and "RTX 5090" in gpu_info["gpu_name"]:
        return True
    return False


def optimize_for_rtx_5090():
    """Optimize settings for RTX 5090 graphics card"""
    try:
        import torch
        if torch.cuda.is_available() and is_rtx_5090():
            # Enable Flash Attention and other optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Set memory allocation strategy
            torch.cuda.empty_cache()
            
            return True
    except Exception:
        pass
    
    return False


def decode_image_dataurl(image_dataurl: str) -> Image.Image:
    image_data = re.sub("^data:image/.+;base64,", "", image_dataurl)
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    return image


def encode_image_dataurl(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(
        "utf-8"
    )
