from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    # Negative prompt doesn't work in LCM
    negative_prompt = ""

    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "stabilityai/stable-diffusion-xl-base-1.0"
        },
        "unet": "latent-consistency/lcm-sdxl",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 9,
            "steps": 4,
            "cfg": 5
        },
        "scheduler": {
            "method": "LCMScheduler"
        }
    }

    images = run_inference_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lcm.png")
