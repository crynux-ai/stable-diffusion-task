from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    args = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": "hamburger,, lettuce, mayo, lettuce, no tomato",
        "task_config": {
            "num_images": 9,
            "steps": 30
        },
        "lora": {
            "model": "ostris/ikea-instructions-lora-sdxl"
        }

    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora_hf.png")
