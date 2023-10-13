from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    args = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
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
            "steps": 30
        },
        "lora": {
            "model": "stillsuit-sdxl"
        }

    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora.png")
