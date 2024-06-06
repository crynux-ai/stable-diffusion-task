from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = ("a realistic photo of an old man sitting on a brown chair, on the seaside, with blue sky and white "
              "clouds, a dog is lying under his legs, masterpiece, high resolution")

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
            "cfg": 7,
            "seed": 99975892,
            "steps": 40
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15.png")
