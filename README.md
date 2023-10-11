## Stable Diffusion Task

A general framework to define and execute the Stable Diffusion task.

### Features

* Same task definition to run Stable Diffusion 1.5, 2.1 and Stable Diffusion XL tasks
* SDXL - Base + Refiner ([ensemble of expert denoisers](https://research.nvidia.com/labs/dir/eDiff-I/)) and standalone refiner
* Controlnet and various preprocessing methods
* LoRA
* VAE

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
(venv) $ python ./prefetch.py
```

Check and run the examples:
```shell
(venv) $ python ./start.py
```

### Task Definition

The complete task definition can be found in the file [```./gen_image_args.py```](./gen_image_args.py)
