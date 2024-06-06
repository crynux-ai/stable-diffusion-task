from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    prompt = ("best quality, ultra high res, (photorealistic:1.4), 1girl, off-shoulder dress, black skirt, "
              "studio background, (faded ash gray hair:1.2), (huge breasts:1.2), looking at viewer, "
              "closeup <lora:JDLv15:0.66>")

    negative_prompt = ("paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, "
                       "normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, "
                       "glans,")

    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "emilianJR/chilloutmix_NiPrunedFp32Fix",
            "variant": None
        },
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 9,
            "safety_checker": False,
            "cfg": 8,
            "seed": 2116617370,
            "steps": 28
        },
        "lora": {
            "model": "https://civitai.com/api/download/models/34562",
            "weight": 70
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_lora_downloaded.png")
