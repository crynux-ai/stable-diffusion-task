import validators
import hashlib
import os
import urllib3
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME
from diffusers.loaders import LORA_WEIGHT_NAME_SAFE, LORA_WEIGHT_NAME, TEXT_INVERSION_NAME_SAFE, TEXT_INVERSION_NAME
from diffusers import DiffusionPipeline

from sd_task.config import ProxyConfig
from tqdm import tqdm
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.utils import EntryNotFoundError
from typing import Callable, Union
from diffusers import AutoencoderKL, ControlNetModel


def check_and_prepare_models(
        task_args: InferenceTaskArgs,
        **kwargs):

    task_args.base_model = check_and_download_hf_pipeline(
        task_args.base_model,
        **kwargs
    )

    if task_args.refiner is not None and task_args.refiner.model != "":
        task_args.refiner.model = check_and_download_hf_pipeline(
            task_args.refiner.model,
            **kwargs
        )

    if task_args.vae != "":
        task_args.vae = check_and_download_model_by_name(
            task_args.vae,
            AutoencoderKL.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs
        )

    if task_args.controlnet is not None:
        task_args.controlnet.model = check_and_download_model_by_name(
            task_args.controlnet.model,
            ControlNetModel.load_config,
            [SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME],
            False,
            **kwargs
        )

    if task_args.lora is not None:
        task_args.lora.model = check_and_download_model_by_name(
            task_args.lora.model,
            None,
            [LORA_WEIGHT_NAME_SAFE, LORA_WEIGHT_NAME],
            True,
            **kwargs
        )

    if task_args.textual_inversion != "":
        task_args.textual_inversion = check_and_download_model_by_name(
            task_args.textual_inversion,
            None,
            [TEXT_INVERSION_NAME_SAFE, TEXT_INVERSION_NAME],
            True,
            **kwargs
        )


def check_and_download_model_by_name(
        model_name: str,
        loader: Union[Callable, None],
        weights_names: list[str],
        guess_weights_name: bool,
        **kwargs) -> str:
    hf_model_cache_dir = kwargs.pop("hf_model_cache_dir")
    external_model_cache_dir = kwargs.pop("external_model_cache_dir")
    proxy = kwargs.pop("proxy")

    if validators.url(model_name):
        return check_and_download_external_model(model_name, external_model_cache_dir, proxy)
    else:
        return check_and_download_hf_model(
            model_name,
            loader,
            weights_names,
            guess_weights_name,
            hf_model_cache_dir,
            proxy)


def check_and_download_external_model(
        model_name: str,
        external_cache_dir: str,
        proxy: ProxyConfig | None
) -> str:

    print("Check and download the external model file: " + model_name)

    m = hashlib.sha256()
    m.update(model_name.encode('utf-8'))
    url_hash = m.hexdigest()

    model_folder = os.path.join(external_cache_dir, url_hash)
    model_file = os.path.join(model_folder, "model.safetensors")

    print("The model file will be saved as: " + model_file)

    # Check if we have already cached the model file
    if os.path.isdir(model_folder):
        if os.path.isfile(model_file):
            print("Found a local cache of the model file. Skip the download")
            return model_file
    else:
        os.mkdir(model_folder, 0o755)

    # Download the model file
    model_file = os.path.join(model_folder, "model.safetensors")

    print("Model file not cached locally. Start the download...")

    try:
        if proxy.host != "":
            proxy_str = proxy.host + ":" + str(proxy.port)

            print("Download using proxy: " + proxy_str)

            default_headers = None
            if proxy.username != "" and proxy.password != "":
                default_headers = urllib3.make_headers(proxy_basic_auth=proxy.username + ':' + proxy.password)
            http = urllib3.ProxyManager(
                proxy_str,
                proxy_headers=default_headers,
                num_pools=1
            )
        else:
            http = urllib3.PoolManager(num_pools=1)

        resp = http.request("GET", model_name, preload_content=False)
        total_bytes = resp.getheader("content-length", None)
        if total_bytes is not None:
            total_bytes = int(total_bytes)

        with tqdm.wrapattr(open(model_file, "wb"), "write",
                           miniters=1, desc=model_file,
                           total=total_bytes) as f_out:
            for chunk in resp.stream(1024):
                if chunk:
                    f_out.write(chunk)
            f_out.flush()

        return model_folder
    except Exception as e:
        # delete the broken file if download failed
        if os.path.isfile(model_file):
            os.remove(model_file)

        raise e


def check_and_download_hf_pipeline(
    model_name: str,
    **kwargs
) -> str:
    print("Check and download the Huggingface pipeline: " + model_name)

    hf_model_cache_dir = kwargs.pop("hf_model_cache_dir")
    proxy = kwargs.pop("proxy")

    call_args = {
        "cache_dir": hf_model_cache_dir,
        "proxies": get_hf_proxy_dict(proxy),
        "resume_download": True
    }
    DiffusionPipeline.download(model_name, **call_args)
    return model_name


def check_and_download_hf_model(
        model_name: str,
        config_loader: Union[Callable, None],
        weights_names: list[str],
        guess_weight_name: bool,
        hf_model_cache_dir: str,
        proxy: ProxyConfig | None

) -> str:
    print("Check and download the Huggingface model file: " + model_name)

    call_args = {
        "cache_dir": hf_model_cache_dir,
        "proxies": get_hf_proxy_dict(proxy),
        "resume_download": True
    }

    # download the config file
    if config_loader is not None:
        config_loader(model_name, **call_args)

    model_file = None

    for weights_name in weights_names:
        if model_file is None:
            try:
                call_args["filename"] = add_variant(weights_name, "fp16")
                model_file = hf_hub_download(model_name, **call_args)
            except EntryNotFoundError:
                pass

    if model_file is None:
        for idx, weights_name in enumerate(weights_names):

            if model_file is None:

                call_args["filename"] = weights_name

                if (not guess_weight_name) and idx == len(weights_names) - 1:
                    model_file = hf_hub_download(model_name, **call_args)
                else:
                    try:
                        model_file = hf_hub_download(model_name, **call_args)
                    except EntryNotFoundError:
                        pass

    if model_file is None and guess_weight_name:
        weight_name = best_guess_weight_name(model_name, file_extension=".safetensors")
        if weight_name is None:
            weight_name = best_guess_weight_name(model_name, file_extension=".bin")

        if weight_name is None:
            # To raise the same EntryNotFound Error
            hf_hub_download(model_name, **call_args)
            return model_name

        call_args["filename"] = weight_name
        hf_hub_download(model_name, **call_args)

    return model_name


def get_hf_proxy_dict(proxy: ProxyConfig | None) -> dict | None:
    if proxy is not None and proxy.host != "":

        proxy_str = proxy.host + ":" + str(proxy.port)

        return {
            'https': proxy_str,
            'http': proxy_str
        }
    else:
        return None


def add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


def best_guess_weight_name(pretrained_model_name_or_path_or_dict, file_extension=".safetensors") -> str | None:

    files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
    targeted_files = [f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)]

    if len(targeted_files) == 0:
        return

    unallowed_substrings = {"scheduler", "optimizer", "checkpoint"}
    targeted_files = list(
        filter(lambda x: all(substring not in x for substring in unallowed_substrings), targeted_files)
    )

    if len(targeted_files) > 1:
        raise ValueError(
            f"Provided path contains more than one weights file in the {file_extension} format. Either specify `weight_name` in `load_lora_weights` or make sure there's only one  `.safetensors` or `.bin` file in  {pretrained_model_name_or_path_or_dict}."
        )
    weight_name = targeted_files[0]
    return weight_name
