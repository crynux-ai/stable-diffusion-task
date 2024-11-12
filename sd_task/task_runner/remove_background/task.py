import os
from typing import cast

import torch
from PIL import Image
from rembg import new_session, remove
from sd_task import utils
from sd_task.cache import ModelCache
from sd_task.config import Config, get_config
from sd_task.task_args.remove_background import RemoveBackgroundTaskArgs


def run_remove_background_task(
    args: RemoveBackgroundTaskArgs,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
):
    if config is None:
        config = get_config()

    if args.image is not None:
        image = args.image
    elif len(args.image_dataurl) > 0:
        image = utils.decode_image_dataurl(args.image_dataurl)
    else:
        raise ValueError("Image and image_dataurl cannot be both empty")

    if config.deterministic and utils.get_accelerator() == "cuda":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

    def load_model():
        os.environ["U2NET_HOME"] = config.data_dir.models.external
        providers = []
        if utils.get_accelerator() == "cuda":
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "cudnn_conv_algo_search": "DEFAULT",
                        "use_tf32": 0,
                        "device_id": torch.cuda.current_device(),
                    },
                )
            )
        providers.append("CPUExecutionProvider")
        session = new_session(args.model, providers=providers)
        return session

    if model_cache is not None:
        session = model_cache.load("remove_bg_session", load_model)
    else:
        session = load_model()

    output = remove(image,
        alpha_matting=args.alpha_matting,
        only_mask=args.return_mask,
        session=session
    )
    output = cast(Image.Image, output)
    return output
