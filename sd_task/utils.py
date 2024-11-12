import base64
import platform
import re
from io import BytesIO

from PIL import Image


def get_accelerator():
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
