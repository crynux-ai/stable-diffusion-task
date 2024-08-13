from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "stabilityai/stable-diffusion-xl-base-1.0"
        },
        "prompt": "hamburger, lettuce, mayo, lettuce, no tomato",
        "task_config": {
            "num_images": 9,
            "steps": 30
        },
        "lora": {
            "model": "ostris/ikea-instructions-lora-sdxl"
        }
    }

    images = run_inference_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora_hf.png")
