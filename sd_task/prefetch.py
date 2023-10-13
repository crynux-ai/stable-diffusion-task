import torch
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    ControlNetModel,
)
import sd_task.config as config


def get_pretrained_args(model_config):
    args = {
        "pretrained_model_name_or_path": model_config["id"],
        "torch_dtype": torch.float16,
        "resume_download": True,
        "cache_dir": config.config["data_dir"]["models"]["huggingface"]
    }

    if "variant" in model_config:
        args["variant"] = model_config["variant"]

    return args


def prefetch_models():

    # base models
    if config.config["preloaded_models"]["base"] is not None:
        for model_config in config.config["preloaded_models"]["base"]:
            print("Preloading base model: ", model_config["id"])

            model_args = get_pretrained_args(model_config)
            DiffusionPipeline.from_pretrained(**model_args)

            print("Successfully preloaded base model: ", model_config["id"])

    # controlnet models
    if config.config["preloaded_models"]["controlnet"] is not None:
        for model_config in config.config["preloaded_models"]["controlnet"]:
            print("Preloading controlnet model: ", model_config["id"])

            model_args = get_pretrained_args(model_config)
            ControlNetModel.from_pretrained(**model_args)

            print("Successfully preloaded controlnet model: ", model_config["id"])

    # vae models
    if config.config["preloaded_models"]["vae"] is not None:
        for model_config in config.config["preloaded_models"]["vae"]:
            print("Preloading vae model: ", model_config["id"])

            model_args = get_pretrained_args(model_config)
            AutoencoderKL.from_pretrained(**model_args)

            print("Successfully preloaded vae model: ", model_config["id"])


if __name__ == "__main__":
    config.load_config()
    prefetch_models()
