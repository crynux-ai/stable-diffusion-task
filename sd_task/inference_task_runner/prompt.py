import re

from compel import Compel, ReturnedEmbeddingsType

from sd_task.inference_task_args.task_args import InferenceTaskArgs


def add_prompt_pipeline_call_args(call_args, pipeline, args: InferenceTaskArgs):
    class_name = type(pipeline).__name__

    if re.match(r"StableDiffusionXL", class_name) is None:
        add_prompt_pipeline_sd15_call_args(call_args, pipeline, args)
    else:
        add_prompt_pipeline_sdxl_call_args(call_args, pipeline, args)


def add_prompt_pipeline_sdxl_call_args(call_args, pipeline, args: InferenceTaskArgs):
    compel = Compel(
        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=[False, True],
        truncate_long_prompts=False,
    )

    conditioning, pooled = compel([args.prompt, args.negative_prompt])

    call_args["prompt_embeds"] = conditioning[0:1]
    call_args["pooled_prompt_embeds"] = pooled[0:1]
    call_args["negative_prompt_embeds"] = conditioning[1:2]
    call_args["negative_pooled_prompt_embeds"] = pooled[1:2]


def add_prompt_refiner_sdxl_call_args(call_args, refiner, args: InferenceTaskArgs):
    compel = Compel(
        tokenizer=refiner.tokenizer_2,
        text_encoder=refiner.text_encoder_2,
        returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        requires_pooled=True,
        truncate_long_prompts=False,
    )

    refiner_prompt_embeds, refiner_pooled_prompt_embeds = compel(args.prompt)
    refiner_negative_prompt_embeds, refiner_negative_pooled_prompt_embeds = compel(
        args.negative_prompt
    )

    call_args["prompt_embeds"] = refiner_prompt_embeds
    call_args["pooled_prompt_embeds"] = refiner_pooled_prompt_embeds
    call_args["negative_prompt_embeds"] = refiner_negative_prompt_embeds
    call_args["negative_pooled_prompt_embeds"] = refiner_negative_pooled_prompt_embeds


def add_prompt_pipeline_sd15_call_args(call_args, pipeline, args: InferenceTaskArgs):
    compel = Compel(
        tokenizer=pipeline.tokenizer,
        text_encoder=pipeline.text_encoder,
        requires_pooled=False,
        truncate_long_prompts=False,
    )

    prompt_embeds = compel.build_conditioning_tensor(args.prompt)

    call_args["prompt_embeds"] = prompt_embeds

    if args.negative_prompt != "":
        neg_prompt_embeds = compel.build_conditioning_tensor(args.negative_prompt)
        call_args["negative_prompt_embeds"] = neg_prompt_embeds
