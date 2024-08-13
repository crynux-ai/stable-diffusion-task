from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = ("best quality, ultra high res, photorealistic++++, 1girl, off-shoulder sweater, smiling, "
              "faded ash gray messy bun hair+, border light, depth of field, looking at "
              "viewer, closeup")

    negative_prompt = ("paintings, sketches, worst quality+++++, low quality+++++, normal quality+++++, lowres, "
                       "normal quality, monochrome++, grayscale++, skin spots, acnes, skin blemishes, "
                       "age spot, glans")

    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "stabilityai/stable-diffusion-xl-base-1.0"
        },
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 9,
            "steps": 40
        },
        "refiner": {
            "model": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "denoising_cutoff": 80
        }
    }

    images = run_inference_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_base_refiner.png")
