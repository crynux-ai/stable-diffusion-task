## Stable Diffusion Task

A general framework to define and execute the Stable Diffusion task.


### Features

* Unified task definition for Stable Diffusion 1.5, 2.1 and Stable Diffusion XL
* SDXL - Base + Refiner ([ensemble of expert denoisers](https://research.nvidia.com/labs/dir/eDiff-I/)) and standalone Refiner
* Controlnet and various preprocessing methods
* LoRA
* VAE
* Textual Inversion
* Long prompt
* Prompt weighting using [Compel](https://github.com/damian0815/compel)
* Maximized reproducibility
* Auto LoRA and Textual Inversion model downloading from non-huggingface URL


### Example

Here is an example of the SDXL image generation, with LoRA, ControlNet and SDXL image refiner:

```python
from sd_task.inference_task_runner.inference_task import run_task
from sd_task.inference_task_args.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    args = {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": "best quality, ultra high res, photorealistic++++, 1girl, desert, full shot, dark stillsuit, "
                  "stillsuit mask up, gloves, solo, highly detailed eyes,"
                  "hyper-detailed, high quality visuals, dim Lighting, ultra-realistic, sharply focused, octane render,"
                  "8k UHD",
        "negative_prompt": "no moon++, buried in sand, bare hands, figerless gloves, "
                           "blue stillsuit, barefoot, weapon, vegetation, clouds, glowing eyes++, helmet, "
                           "bare handed, no gloves, double mask, simplified, abstract, unrealistic, impressionistic, "
                           "low resolution,",
        "task_config": {
            "num_images": 9,
            "steps": 30
        },
        "lora": {
            "model": "https://civitai.com/api/download/models/178048"
        },
        "controlnet": {
            "model": "diffusers/controlnet-canny-sdxl-1.0",
            "image_dataurl": "data:image/png;base64,12FE1373...",
            "preprocess": {
                "method": "canny"
            },
            "weight": 70
        },
        "refiner": {
            "model": "stabilityai/stable-diffusion-xl-refiner-1.0"
        }
    }

    images = run_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora_controlnet_refiner.png")
```


### Get started

Create and activate the virtual environment:
```shell
$ python -m venv ./venv
$ source ./venv/bin/activate
```

Install the dependencies:
```shell
(venv) $ pip install -r requirments.txt
```

Cache the base model files:
```shell
(venv) $ python ./sd_task/prefetch.py
```

Check and run the examples:
```shell
(venv) $ python ./examples/sd15_controlnet_openpose.py
```

More explanations can be found in the doc:

[https://docs.crynux.ai/application-development/stable-diffusion-task](https://docs.crynux.ai/application-development/stable-diffusion-task)

### Task Definition

The complete task definition can be found in the file [```./sd_task/inference_task_args/task_args.py```](sd_task/inference_task_args/task_args.py)

### JSON Schema

The JSON schemas for the tasks could be used to validate the task arguments by other projects.
The schemas are given under [```./schema```](./schema). Projects could use the URL to load the JSON schema files directly.
