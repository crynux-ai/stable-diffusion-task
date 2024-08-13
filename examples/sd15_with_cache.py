from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from sd_task.cache import MemoryModelCache

if __name__ == '__main__':
    cache = MemoryModelCache()

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
            "num_images": 1,
            "safety_checker": False,
            "cfg": 7,
            "seed": 99975892,
            "steps": 40
        }
    }

    for i in range(5):
        images = run_inference_task(InferenceTaskArgs.model_validate(args), model_cache=cache)
        images[0].save(f"./data/sd15_{i}.png")
