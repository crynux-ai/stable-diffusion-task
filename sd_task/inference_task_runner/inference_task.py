import random
import numpy as np
import os
from typing import List, Dict, Any

import torch
from diffusers import (AutoencoderKL, AutoPipelineForText2Image,
                       ControlNetModel, DiffusionPipeline,
                       DPMSolverMultistepScheduler)
from PIL import Image

from sd_task.config import get_config, Config
from sd_task.inference_task_args.task_args import InferenceTaskArgs

from .controlnet import add_controlnet_pipeline_call_args
from .prompt import (add_prompt_pipeline_call_args,
                     add_prompt_refiner_sdxl_call_args)

from .download_model import check_and_prepare_models

from .errors import TaskExecutionError, ModelDownloadError

from .log import log

# Use deterministic algorithms for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def get_pipeline_init_args(cache_dir: str, args: InferenceTaskArgs | None = None):
    init_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "cache_dir": cache_dir,
        "local_files_only": True
    }

    if args is not None and args.task_config.safety_checker is False:
        init_args["safety_checker"] = None

    return init_args


def prepare_pipeline(cache_dir: str, args: InferenceTaskArgs):
    pipeline_args = get_pipeline_init_args(cache_dir, args)

    if args.controlnet is not None:
        controlnet_model = None
        try:
            controlnet_model = ControlNetModel.from_pretrained(
                args.controlnet.model,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                variant="fp16",
                local_files_only=True
            )
        except EnvironmentError:
            pass

        if controlnet_model is None:
            controlnet_model = ControlNetModel.from_pretrained(
                args.controlnet.model,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                local_files_only=True
            )

        pipeline_args["controlnet"] = controlnet_model.to("cuda")

    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.base_model, **pipeline_args
    )

    # Faster scheduler from the huggingface doc, requires only ~20-25 steps
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    if args.vae != "":
        vae_model = None
        try:
            vae_model = AutoencoderKL.from_pretrained(
                args.vae,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                local_files_only=True,
                variant="fp16",
            )
        except EnvironmentError:
            pass

        if vae_model is None:
            vae_model = AutoencoderKL.from_pretrained(
                args.vae,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                local_files_only=True,
            )

        pipeline.vae = vae_model.to("cuda")

    if args.lora is not None:
        # raises ValueError if the lora model is not compatible with the base model
        pipeline.load_lora_weights(
            args.lora.model,
            lora_scale=args.lora.weight,
            cache_dir=cache_dir,
            local_files_only=True
        )

    if args.textual_inversion != "":
        pipeline.load_textual_inversion(
            args.textual_inversion,
            cache_dir=cache_dir,
            local_files_only=True
        )

    pipeline = pipeline.to("cuda")

    # Refiner pipeline
    refiner = None

    if args.refiner is not None:
        refiner_init_args = get_pipeline_init_args(cache_dir)
        refiner_init_args["tokenizer_2"] = pipeline.tokenizer_2
        refiner_init_args["text_encoder_2"] = pipeline.text_encoder_2
        refiner_init_args["vae"] = pipeline.vae
        refiner = DiffusionPipeline.from_pretrained(
            args.refiner.model, **refiner_init_args
        ).to("cuda")

    return pipeline, refiner


def get_pipeline_call_args(pipeline, args: InferenceTaskArgs) -> Dict[str, Any]:
    call_args: Dict[str, Any] = {
        "num_inference_steps": args.task_config.steps,
        "width": args.task_config.image_width,
        "height": args.task_config.image_height,
        "guidance_scale": args.task_config.cfg,

        # generator on CPU for reproducibility
        "generator": torch.Generator(device="cpu").manual_seed(args.task_config.seed)
    }

    add_prompt_pipeline_call_args(call_args, pipeline, args)

    if args.controlnet is not None:
        add_controlnet_pipeline_call_args(call_args, args)

    if args.refiner is not None:
        call_args["output_type"] = "latent"

        # denoising_end is not supported by StableDiffusionXLControlNetPipeline yet.
        if args.controlnet is None:
            call_args["denoising_end"] = args.refiner.denoising_cutoff

    return call_args


def run_task(args: InferenceTaskArgs, config: Config | None = None) -> List[Image.Image]:

    if config is None:
        config = get_config()

    log("Task is started")

    log("Check the model cache and download the models")
    try:
        check_and_prepare_models(
            args,
            external_model_cache_dir=config.data_dir.models.external,
            hf_model_cache_dir=config.data_dir.models.huggingface,
            proxy=config.proxy
        )
    except Exception as e:
        raise ModelDownloadError() from e

    log("All the required models are downloaded")

    torch.manual_seed(args.task_config.seed)
    random.seed(args.task_config.seed)
    np.random.seed(args.task_config.seed)

    try:
        log("Start loading the pipeline")
        pipeline, refiner = prepare_pipeline(config.data_dir.models.huggingface, args)

        generated_images = []

        call_args = get_pipeline_call_args(pipeline, args)

        refiner_call_args = {
            "generator": call_args["generator"]
        }

        if args.refiner is not None and refiner is not None:
            add_prompt_refiner_sdxl_call_args(refiner_call_args, refiner, args)
            refiner_call_args["num_inference_steps"] = args.refiner.steps

            # denoising_end is not supported by StableDiffusionXLControlNetPipeline yet.
            if args.controlnet is None:
                refiner_call_args["denoising_start"] = args.refiner.denoising_cutoff

        log("The pipeline has been successfully loaded")

        log("The images generation is started")
        for i in range(args.task_config.num_images):
            image = pipeline(**call_args)

            if refiner is not None:
                refiner_call_args["image"] = image.images
                image = refiner(**refiner_call_args)

            generated_images.append(image.images[0])

        log("The images generation is finished")

        return generated_images
    except Exception as e:
        raise TaskExecutionError() from e
