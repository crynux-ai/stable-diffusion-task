from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from sd_task.config import load_config
from diffusers.utils import make_image_grid
from reference_image import get_controlnet_ref_image_dataurl


if __name__ == '__main__':
    load_config()

    ref_image = get_controlnet_ref_image_dataurl()

    prompt = ("best quality, ultra high res, photorealistic++++, 1girl, off-shoulder sweate, smiling, "
              "faded ash gray messy bun hair+, border light, depth of field, looking at "
              "viewer, closeup")

    negative_prompt = ("paintings, sketches, worst quality+++++, low quality+++++, normal quality+++++, lowres, "
                       "normal quality, monochrome++, grayscale++, skin spots, acnes, skin blemishes, "
                       "age spot, glans")

    args = {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 9,
            "safety_checker": False
        },
        "controlnet": {
            "model": "lllyasviel/sd_controlnet_canny",
            "image_dataurl": ref_image,
            "preprocess": {
                "method": "canny",
                "args": {
                    "low_threshold": 50,
                    "high_threshold": 100
                }
            }
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_controlnet_canny.png")
