import math
import os
from typing import List

import torch
from diffusers import (AutoencoderKL, AutoPipelineForText2Image,
                       ControlNetModel, DiffusionPipeline,
                       DPMSolverMultistepScheduler)
from PIL import Image

from sd_task import config
from sd_task.inference_task_args.task_args import InferenceTaskArgs

from .controlnet import add_controlnet_pipeline_call_args
from .prompt import (add_prompt_pipeline_call_args,
                     add_prompt_refiner_sdxl_call_args)

# Use deterministic algorithms for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def get_pipeline_init_args(args: InferenceTaskArgs = None):
    init_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "local_files_only": True,
        "cache_dir": config.config["data_dir"]["models"]["huggingface"],
    }

    if args is not None and args.task_config.safety_checker is False:
        init_args["safety_checker"] = None

    return init_args


def prepare_pipeline(args: InferenceTaskArgs):
    pipeline_args = get_pipeline_init_args(args)

    if args.controlnet is not None:
        controlnet_model = ControlNetModel.from_pretrained(
            args.controlnet.model,
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir=config.config["data_dir"]["models"]["huggingface"],
        )

        pipeline_args["controlnet"] = controlnet_model

    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.base_model, **pipeline_args
    )

    # Faster scheduler from the huggingface doc, requires only ~20-25 steps
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )

    if args.vae != "":
        pipeline.vae = AutoencoderKL.from_pretrained(
            args.vae,
            torch_dtype=torch.float16,
            local_files_only=True,
            cache_dir=config.config["data_dir"]["models"]["huggingface"],
        ).to("cuda")

    if args.lora is not None:
        # raises ValueError if the lora model is not compatible with the base model
        pipeline.load_lora_weights(
            os.path.join(config.config["data_dir"]["models"]["lora"], args.lora.model),
            lora_scale=args.lora.weight,
            local_files_only=True,
        )

    if args.textual_inversion != "":
        pipeline.load_textual_invertion(
            args.textual_inversion,
            local_files_only=True,
            cache_dir=config.config["data_dir"]["models"]["huggingface"],
        )

    pipeline = pipeline.to("cuda")

    # Refiner pipeline
    refiner = None

    if args.refiner is not None:
        refiner_init_args = get_pipeline_init_args()
        refiner_init_args["tokenizer_2"] = pipeline.tokenizer_2
        refiner_init_args["text_encoder_2"] = pipeline.text_encoder_2
        refiner_init_args["vae"] = pipeline.vae
        refiner = DiffusionPipeline.from_pretrained(
            args.refiner.model, **refiner_init_args
        ).to("cuda")

    return pipeline, refiner


def get_pipeline_call_args(pipeline, args: InferenceTaskArgs):
    call_args = {
        "num_inference_steps": args.task_config.steps,
        "width": args.task_config.image_width,
        "height": args.task_config.image_height,
        "guidance_scale": args.task_config.cfg,
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


def run_task(args: InferenceTaskArgs) -> List[Image.Image]:
    pipeline, refiner = prepare_pipeline(args)

    generated_images = []

    call_args = get_pipeline_call_args(pipeline, args)
    refiner_call_args = {}

    if refiner is not None:
        add_prompt_refiner_sdxl_call_args(refiner_call_args, refiner, args)
        refiner_call_args["num_inference_steps"] = math.ceil(args.refiner.steps)

        # denoising_end is not supported by StableDiffusionXLControlNetPipeline yet.
        if args.controlnet is None:
            refiner_call_args["denoising_start"] = args.refiner.denoising_cutoff

    for i in range(args.task_config.num_images):
        current_seed = args.task_config.seed + i * 3000

        # generator on CPU for reproducibility
        call_args["generator"] = torch.manual_seed(current_seed)

        image = pipeline(**call_args)

        if refiner is not None:
            refiner_call_args["image"] = image.images
            refiner_call_args["generator"] = torch.manual_seed(current_seed)
            image = refiner(**refiner_call_args)

        generated_images.append(image.images[0])

    return generated_images
