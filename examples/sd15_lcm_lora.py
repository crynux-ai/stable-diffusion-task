from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
    negative_prompt = ""

    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "runwayml/stable-diffusion-v1-5"
        },
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 9,
            "safety_checker": False,
            "steps": 4,
            "cfg": 1
        },
        "lora": {
            "model": "latent-consistency/lcm-lora-sdv1-5",
            "weight": 70
        },
        "scheduler": {
            "method": "LCMScheduler"
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_lcm_lora.png")
