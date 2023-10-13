from __future__ import annotations

import torch
from diffusers import AutoencoderKL, ControlNetModel, DiffusionPipeline

from sd_task.config import Config, ModelConfig, get_config


def get_pretrained_args(model_config: ModelConfig, cache_dir: str):
    args = {
        "pretrained_model_name_or_path": model_config.id,
        "torch_dtype": torch.float16,
        "resume_download": True,
        "cache_dir": cache_dir
    }

    if model_config.variant is not None:
        args["variant"] = model_config.variant

    return args


def prefetch_models(config: Config | None = None):
    if config is None:
        config = get_config()

    # base models
    if config.preloaded_models.base is not None:
        for model_config in config.preloaded_models.base:
            print("Preloading base model: ", model_config.id)

            model_args = get_pretrained_args(model_config, config.data_dir.models.huggingface)
            DiffusionPipeline.from_pretrained(**model_args)

            print("Successfully preloaded base model: ", model_config.id)

    # controlnet models
    if config.preloaded_models.controlnet is not None:
        for model_config in config.preloaded_models.controlnet:
            print("Preloading controlnet model: ", model_config.id)

            model_args = get_pretrained_args(model_config, config.data_dir.models.huggingface)
            ControlNetModel.from_pretrained(**model_args)

            print("Successfully preloaded controlnet model: ", model_config.id)

    # vae models
    if config.preloaded_models.vae is not None:
        for model_config in config.preloaded_models.vae:
            print("Preloading vae model: ", model_config.id)

            model_args = get_pretrained_args(model_config, config.data_dir.models.huggingface)
            AutoencoderKL.from_pretrained(**model_args)

            print("Successfully preloaded vae model: ", model_config)


if __name__ == "__main__":
    prefetch_models()
