from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid
from reference_image import get_controlnet_ref_image_dataurl


if __name__ == '__main__':
    ref_image = get_controlnet_ref_image_dataurl()

    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "stabilityai/sdxl-turbo"
        },
        "prompt": "best quality, ultra high res, photorealistic++++, 1girl, desert, full shot, dark stillsuit, "
                  "stillsuit mask up, gloves, solo, highly detailed eyes,"
                  "hyper-detailed, high quality visuals, dim Lighting, ultra-realistic, sharply focused, octane render,"
                  "8k UHD",
        "negative_prompt": "no moon++, buried in sand, bare hands, figerless gloves, "
                           "blue stillsuit, barefoot, weapon, vegetation, clouds, glowing eyes++, helmet, "
                           "bare handed, no gloves, double mask, simplified, abstract, unrealistic, impressionistic, "
                           "low resolution,",
        "task_config": {
            "num_images": 9,
            "cfg": 0,
            "steps": 1
        },

        "lora": {
            "model": "https://civitai.com/api/download/models/178048"
        },
        "controlnet": {
            "model": "diffusers/controlnet-canny-sdxl-1.0",
            "image_dataurl": ref_image,
            "preprocess": {
                "method": "canny"
            },
            "weight": 70
        },
        "scheduler": {
            "method": "EulerAncestralDiscreteScheduler",
            "args": {
                "timestep_spacing": "trailing"
            }
        }
    }

    images = run_inference_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_turbo_lora_controlnet.png")
