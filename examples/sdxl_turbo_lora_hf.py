from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = "toy_face of a hacker with a hoodie, pixel art"
    negative_prompt = ""

    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "stabilityai/sdxl-turbo"
        },
        "scheduler": {
            "method": "EulerAncestralDiscreteScheduler",
            "args": {
                "timestep_spacing": "trailing"
            }
        },
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 9,
            "steps": 1,
            "cfg": 0
        },
        "lora": {
            "model": "CiroN2022/toy-face"
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_turbo_lora_hf.png")
