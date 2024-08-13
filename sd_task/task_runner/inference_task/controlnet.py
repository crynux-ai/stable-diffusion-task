import base64
import re
from io import BytesIO

from PIL import Image

from sd_task.task_args.inference_task import ControlnetArgs


def add_controlnet_pipeline_call_args(call_args: dict, controlnet: ControlnetArgs, image_width: int, image_height: int):
    image_data = re.sub(
        '^data:image/.+;base64,',
        '',
        controlnet.image_dataurl)

    reference_image = Image.open(BytesIO(base64.b64decode(image_data)))

    if controlnet.preprocess is not None:
        from controlnet_aux import processor

        args_dict = {}

        if (hasattr(controlnet.preprocess, 'args')
                and controlnet.preprocess.args is not None):
            args_dict = controlnet.preprocess.args.model_dump()

        args_dict["detect_resolution"] = min(reference_image.width, reference_image.height)
        args_dict["image_resolution"] = min(
            image_width,
            image_height
        )

        preprocessor = processor.Processor(controlnet.preprocess.method, args_dict)
        reference_image = preprocessor(reference_image, to_pil=True)

    call_args["image"] = reference_image
    call_args["controlnet_conditioning_scale"] = controlnet.weight

    return call_args
