import hashlib
import json
from typing import Any, Dict

from sd_task.inference_task_args.task_args import InferenceTaskArgs, BaseModelArgs


def generate_model_key(args: InferenceTaskArgs) -> str:
    assert isinstance(args.base_model, BaseModelArgs)

    model_args: Dict[str, Any] = {
        "base_model": args.base_model.name,
        "textual_inversion": args.textual_inversion,
        "safety_checker": args.task_config.safety_checker,
    }

    if args.base_model.variant is not None:
        model_args["base_model_variant"] = args.base_model.variant
    if args.unet is not None and args.unet != "":
        model_args["unet"] = args.unet
    if args.lora is not None and args.lora.model != "":
        model_args["lora_model_name"] = args.lora.model
        model_args["lora_weight"] = args.lora.weight
    if args.controlnet is not None and args.controlnet != "":
        model_args["controlnet_model_name"] = args.controlnet.model
    if args.vae is not None and args.vae != "":
        model_args["vae"] = args.vae
    if args.refiner is not None and args.refiner.model != "":
        model_args["refiner_model_name"] = args.refiner.model

    model_args_str = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    key = hashlib.md5(model_args_str.encode("utf-8")).hexdigest()
    return key
