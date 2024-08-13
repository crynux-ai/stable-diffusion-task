from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME

from sd_task.download_model import (check_and_download_hf_pipeline,
                                    check_and_download_model_by_name)
from sd_task.task_args.inference_task import (BaseModelArgs, InferenceTaskArgs)


def check_and_prepare_models(task_args: InferenceTaskArgs, **kwargs):

    assert isinstance(task_args.base_model, BaseModelArgs)
    task_args.base_model.name = check_and_download_hf_pipeline(
        task_args.base_model.name, task_args.base_model.variant, **kwargs
    )

    if task_args.refiner is not None and task_args.refiner.model != "":
        task_args.refiner.model = check_and_download_hf_pipeline(
            task_args.refiner.model, task_args.refiner.variant, **kwargs
        )

    if task_args.unet is not None and task_args.unet != "":
        task_args.unet, _ = check_and_download_model_by_name(
            task_args.unet,
            UNet2DConditionModel.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs,
        )

    if task_args.vae != "":
        task_args.vae, _ = check_and_download_model_by_name(
            task_args.vae,
            AutoencoderKL.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs,
        )

    if task_args.controlnet is not None:
        task_args.controlnet.model, _ = check_and_download_model_by_name(
            task_args.controlnet.model,
            ControlNetModel.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs,
        )

    if task_args.lora is not None:
        task_args.lora.model, task_args.lora.weight_file_name = (
            check_and_download_model_by_name(
                task_args.lora.model, None, [], True, **kwargs
            )
        )

    if task_args.textual_inversion != "":
        task_args.textual_inversion, _ = check_and_download_model_by_name(
            task_args.textual_inversion, None, [], True, **kwargs
        )
