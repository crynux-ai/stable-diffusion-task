from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from sd_task.cache import MemoryModelCache

if __name__ == '__main__':
    cache = MemoryModelCache()

    prompt = ("a realistic photo of an old man sitting on a brown chair, on the seaside, with blue sky and white "
              "clouds, a dog is lying under his legs, masterpiece, high resolution")

    negative_prompt = ""

    args = {
        "base_model": "runwayml/stable-diffusion-v1-5",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "task_config": {
            "num_images": 1,
            "safety_checker": False,
            "cfg": 7,
            "seed": 99975892,
            "steps": 40
        }
    }

    for i in range(5):
        images = run_task(InferenceTaskArgs.model_validate(args), model_cache=cache)
        images[0].save(f"./data/sd15_{i}.png")
