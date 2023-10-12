from inference_task_args.task_args import InferenceTaskArgs
import re
from PIL import Image
from io import BytesIO
import base64
from controlnet_aux import processor


def add_controlnet_pipeline_call_args(call_args: dict, gen_image_args: InferenceTaskArgs):
    image_data = re.sub(
        '^data:image/.+;base64,',
        '',
        gen_image_args.controlnet.image_dataurl)

    reference_image = Image.open(BytesIO(base64.b64decode(image_data)))

    if gen_image_args.controlnet.preprocess is not None:

        args_dict = {}

        if (hasattr(gen_image_args.controlnet.preprocess, 'args')
                and gen_image_args.controlnet.preprocess.args is not None):
            args_dict = gen_image_args.controlnet.preprocess.args.model_dump()

        args_dict["detect_resolution"] = min(reference_image.width, reference_image.height)
        args_dict["image_resolution"] = min(
            gen_image_args.task_config.image_width,
            gen_image_args.task_config.image_height
        )

        preprocessor = processor.Processor(gen_image_args.controlnet.preprocess.method, args_dict)
        reference_image = preprocessor(reference_image, to_pil=True)

    call_args["image"] = reference_image
    call_args["controlnet_conditioning_scale"] = gen_image_args.controlnet.weight

    return call_args
