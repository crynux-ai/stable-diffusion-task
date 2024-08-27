import logging

from sd_task.task_args.inference_task import ControlnetArgs
from sd_task.utils import decode_image_dataurl, encode_image_dataurl


_logger = logging.getLogger(__name__)


def add_controlnet_pipeline_call_args(call_args: dict, controlnet: ControlnetArgs, image_width: int, image_height: int):
    if controlnet.image is not None:
        reference_image = controlnet.image
    elif len(controlnet.image_dataurl) > 0:
        reference_image = decode_image_dataurl(controlnet.image_dataurl)
    else:
        raise ValueError("Image and image_dataurl cannot be both empty")

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
        _logger.debug(f"controlnet input imaage: {encode_image_dataurl(reference_image)}")

    call_args["image"] = reference_image
    call_args["controlnet_conditioning_scale"] = controlnet.weight

    return call_args
