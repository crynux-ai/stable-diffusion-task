import re
from typing import Callable, Dict


from sd_task import utils
from sd_task.cache import ModelCache
from sd_task.config import Config, get_config
from sd_task.task_args.tagger import TaggerTaskArgs

from .interrogator import (Interrogator, MLDanbooruInterrogator,
                           WaifuDiffusionInterrogator)

_interrogators: Dict[str, Callable[[], Interrogator]] = {
    "WD14 ViT v1": lambda: WaifuDiffusionInterrogator(
        "WD14 ViT v1", repo_id="SmilingWolf/wd-v1-4-vit-tagger"
    ),
    "WD14 ViT v2": lambda: WaifuDiffusionInterrogator(
        "WD14 ViT v2",
        repo_id="SmilingWolf/wd-v1-4-vit-tagger-v2",
    ),
    "WD14 ConvNeXT v1": lambda: WaifuDiffusionInterrogator(
        "WD14 ConvNeXT v1", repo_id="SmilingWolf/wd-v1-4-convnext-tagger"
    ),
    "WD14 ConvNeXT v2": lambda: WaifuDiffusionInterrogator(
        "WD14 ConvNeXT v2",
        repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
    ),
    "WD14 ConvNeXTV2 v1": lambda: WaifuDiffusionInterrogator(
        "WD14 ConvNeXTV2 v1",
        # the name is misleading, but it's v1
        repo_id="SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    ),
    "WD14 SwinV2 v1": lambda: WaifuDiffusionInterrogator(
        "WD14 SwinV2 v1",
        # again misleading name
        repo_id="SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    ),
    "WD14 moat tagger v2": lambda: WaifuDiffusionInterrogator(
        "WD14 moat tagger v2", repo_id="SmilingWolf/wd-v1-4-moat-tagger-v2"
    ),
    "ML-Danbooru Caformer dec-5-97527": lambda: MLDanbooruInterrogator(
        "ML-Danbooru Caformer dec-5-97527",
        repo_id="deepghs/ml-danbooru-onnx",
        model_path="ml_caformer_m36_dec-5-97527.onnx",
    ),
    "ML-Danbooru TResNet-D 6-30000": lambda: MLDanbooruInterrogator(
        "ML-Danbooru TResNet-D 6-30000",
        repo_id="deepghs/ml-danbooru-onnx",
        model_path="TResnet-D-FLq_ema_6-30000.onnx",
    ),
}

DEFAULT_KAMOJIS = "0_0, (o)_(o), +_+, +_-, ._., <o>_<o>, <|>_<|>, =_=, >_<, 3_3, 6_9, >_o, @_@, ^_^, o_o, u_u, x_x, |_|, ||_||"  # pylint: disable=line-too-long # noqa: E501
kamojis = [x.strip() for x in DEFAULT_KAMOJIS.split(",") if x]


def correct_tag(args: TaggerTaskArgs, tag: str) -> str:
    if tag not in kamojis:
        tag = tag.replace("_", " ")

    if len(args.search_tags) == len(args.replace_tags):
        for i, regex in enumerate(args.search_tags):
            if re.match(regex, tag):
                tag = re.sub(regex, args.replace_tags[i], tag)
                break

    return tag


def is_excluded(args: TaggerTaskArgs, tag: str):
    return any(re.match(regex, tag) for regex in args.exclude_tags)


def finalize(args: TaggerTaskArgs, tags: Dict[str, float], ratings: Dict[str, float]):
    result_tags: Dict[str, float] = {}
    discarded_tags: Dict[str, float] = {}

    _tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
    _ratings = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    ratings = {k: v for k, v in _ratings}

    for tag in args.additional_tags:
        result_tags[tag] = 1.0

    max_cnt = args.count_threshold - len(args.additional_tags)
    count = 0
    for tag, val in _tags:
        if count < max_cnt:
            tag = correct_tag(args, tag)
            if tag not in args.keep_tags:
                if is_excluded(args, tag) or val < args.threshold:
                    if (
                        tag not in args.additional_tags
                        and len(discarded_tags) < max_cnt
                    ):
                        discarded_tags[tag] = val
                    continue
            
            count += 1
            if tag not in args.additional_tags:
                result_tags[tag] = val

    return ratings, result_tags, discarded_tags


def run_tagger_task(
    args: TaggerTaskArgs,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
):
    if config is None:
        config = get_config()

    if args.model not in _interrogators:
        raise ValueError(f"Unsupported tagger model name: {args.model}")

    if args.image is not None:
        image = args.image
    elif len(args.image_dataurl) > 0:
        image = utils.decode_image_dataurl(args.image_dataurl)
    else:
        raise ValueError("Image and image_dataurl cannot be both empty")

    def load_model():
        interrogator = _interrogators[args.model]()
        cache_dir = config.data_dir.models.huggingface
        interrogator.load(cache_dir=cache_dir)
        return interrogator

    if model_cache is not None:
        interrogator = model_cache.load(args.model, load_model)
    else:
        interrogator = load_model()

    ratings, tags = interrogator.interrogate(image)

    ratings, tags, discarded_tags = finalize(args, tags, ratings)
    return ratings, tags, discarded_tags
