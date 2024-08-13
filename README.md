## Stable Diffusion Task

A general framework to define and execute the Stable Diffusion task.


### Features

* The latest acceleration tech to generate images in only 1 step using SDXL Turbo & Latent Consistency Models (LCM)
* Unified task definition for Stable Diffusion 1.5, 2.1 and Stable Diffusion XL
* SDXL - Base + Refiner ([ensemble of expert denoisers](https://research.nvidia.com/labs/dir/eDiff-I/)) and standalone Refiner
* Controlnet and various preprocessing methods
* UNet replacement
* Scheduler configuration
* LoRA
* VAE
* Textual Inversion
* Long prompt
* Prompt weighting using [Compel](https://github.com/damian0815/compel)
* Maximized reproducibility
* Auto LoRA and Textual Inversion model downloading from non-huggingface URL


### Example

Here is an example of the SDXL image generation, with LoRA, ControlNet, 
utilizing SDXL Turbo to generate images in only 1 step.

```python
from sd_task.task_runner import run_inference_task
from sd_task.task_args import InferenceTaskArgs
from diffusers.utils import make_image_grid

if __name__ == '__main__':
    args = {
        "version": "2.0.0",
        "base_model": {
            "name": "stabilityai/sdxl-turbo"
        },
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
            "steps": 1,
            "cfg": 0
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
        "scheduler": {
            "method": "EulerAncestralDiscreteScheduler",
            "args": {
                "timestep_spacing": "trailing"
            }
        }
    }

    images = run_inference_task(InferenceTaskArgs.model_validate(args))
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_turbo_lora_controlnet.png")
```
More examples can be found in [Examples](./examples)

### Get started

Create and activate the virtual environment:
```shell
$ python -m venv ./venv
$ source ./venv/bin/activate
```

Install the dependencies:
```shell
# Use requirements_macos.txt on Macos
(venv) $ pip install -r requirments_cuda.txt
```

Cache the base model files:
```shell
(venv) $ python ./sd_task/prefetch.py
```

Check and run the examples:
```shell
(venv) $ python ./examples/sdxl_turbo_lora_controlnet.py
```

More explanations can be found in the doc:

[https://docs.crynux.ai/application-development/stable-diffusion-task](https://docs.crynux.ai/application-development/stable-diffusion-task)

### Task Definition

The complete task definition can be found in the file [```./sd_task/inference_task_args/task_args.py```](sd_task/inference_task_args/task_args.py)

### JSON Schema

The JSON schemas for the tasks could be used to validate the task arguments by other projects.
The schemas are given under [```./schema```](./schema). Projects could use the URL to load the JSON schema files directly.

***Update JSON schema file***
```bash
# In the root folder of the project

$ ./venv/bin/activate

(venv) $ pip install -r requirements_cuda.txt
(venv) $ pip install .
(venv) $ python ./sd_task/inference_task_args/generate_json_schema.py
```
