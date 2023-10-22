from sd_task.config import Config, get_config
from sd_task.inference_task_runner.download_model import check_and_download_hf_model, check_and_download_hf_pipeline
from diffusers import ControlNetModel, AutoencoderKL
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


def prefetch_models(config: Config | None = None):
    if config is None:
        config = get_config()

    # base models
    if config.preloaded_models.base is not None:
        for model_config in config.preloaded_models.base:
            print("Preloading base model: ", model_config.id)

            call_args = {
                "hf_model_cache_dir": config.data_dir.models.huggingface,
                "proxy": config.proxy
            }

            check_and_download_hf_pipeline(model_config.id, **call_args)
            print("Successfully preloaded base model: ", model_config.id)

    # controlnet models
    if config.preloaded_models.controlnet is not None:
        for model_config in config.preloaded_models.controlnet:
            print("Preloading controlnet model: ", model_config.id)
            check_and_download_hf_model(
                model_config.id,
                ControlNetModel.load_config,
                [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
                False,
                config.data_dir.models.huggingface,
                config.proxy
            )
            print("Successfully preloaded controlnet model: ", model_config.id)

    # vae models
    if config.preloaded_models.vae is not None:
        for model_config in config.preloaded_models.vae:
            print("Preloading vae model: ", model_config.id)
            check_and_download_hf_model(
                model_config.id,
                AutoencoderKL.load_config,
                [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
                False,
                config.data_dir.models.huggingface,
                config.proxy
            )
            print("Successfully preloaded vae model: ", model_config)


if __name__ == "__main__":
    prefetch_models()
