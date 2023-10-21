from __future__ import annotations

from sd_task.config import Config, ModelConfig, ProxyConfig, get_config
from huggingface_hub import snapshot_download


def get_pretrained_args(model_config: ModelConfig, cache_dir: str, proxy: ProxyConfig | None = None):
    args = {
        "repo_id": model_config.id,
        "resume_download": True,
        "cache_dir": cache_dir
    }

    if proxy is not None:
        args["proxies"] = get_hf_proxy_dict(proxy)

    return args


def prefetch_models(config: Config | None = None):
    if config is None:
        config = get_config()

    # base models
    if config.preloaded_models.base is not None:
        for model_config in config.preloaded_models.base:
            print("Preloading base model: ", model_config.id)

            model_args = get_pretrained_args(model_config, config.data_dir.models.huggingface, config.proxy)
            snapshot_download(**model_args)

            print("Successfully preloaded base model: ", model_config.id)

    # controlnet models
    if config.preloaded_models.controlnet is not None:
        for model_config in config.preloaded_models.controlnet:
            print("Preloading controlnet model: ", model_config.id)

            model_args = get_pretrained_args(model_config, config.data_dir.models.huggingface, config.proxy)
            snapshot_download(**model_args)

            print("Successfully preloaded controlnet model: ", model_config.id)

    # vae models
    if config.preloaded_models.vae is not None:
        for model_config in config.preloaded_models.vae:
            print("Preloading vae model: ", model_config.id)

            model_args = get_pretrained_args(model_config, config.data_dir.models.huggingface, config.proxy)
            snapshot_download(**model_args)

            print("Successfully preloaded vae model: ", model_config)


def get_hf_proxy_dict(proxy: ProxyConfig | None) -> dict | None:
    if proxy is not None and proxy.host != "":

        proxy_str = proxy.host + ":" + str(proxy.port)

        return {
            'https': proxy_str,
            'http': proxy_str
        }
    else:
        return None


if __name__ == "__main__":
    prefetch_models()
