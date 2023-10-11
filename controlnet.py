from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import torch
from gen_image_args import GenImageArgs
from controlnet_args import ControlnetArgs, CannyArgs
import config
import re
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np


def canny_preprocess(reference_image: Image, canny_args: CannyArgs):

    high = 200
    low = 100

    if canny_args is not None:
        high = canny_args.high_threshold
        low = canny_args.low_threshold

    image = cv2.Canny(np.array(reference_image), low, high)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def add_controlnet_pipeline_call_args(call_args: dict, controlnet_args: ControlnetArgs):
    image_data = re.sub(
        '^data:image/.+;base64,',
        '',
        controlnet_args.image_dataurl)

    reference_image = Image.open(BytesIO(base64.b64decode(image_data)))

    if controlnet_args.preprocess is not None:
        if controlnet_args.preprocess.method == "canny":
            reference_image = canny_preprocess(reference_image, controlnet_args.preprocess.args)

    call_args["image"] = reference_image
    call_args["controlnet_conditioning_scale"] = controlnet_args.weight

    return call_args
