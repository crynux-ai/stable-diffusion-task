from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = "a grafitti in a favela wall with a <cat-toy> on it"

    args = {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "prompt": prompt,
        "textual_inversion": "sd-concepts-library/cat-toy",
        "task_config": {
            "num_images": 9
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_textual_inversion.png")
