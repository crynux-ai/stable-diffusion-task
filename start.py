from controlnet_args import ControlnetArgs, PreprocessArgs
from diffusers.utils import make_image_grid
from config import load_config
from PIL import Image
import io
import base64
from gen_image_args import GenImageArgs, LoraArgs, TaskConfig, RefinerArgs
from gen_images import gen_images

prompt = ("best quality, ultra high res, photorealistic++++, 1girl, off-shoulder sweate, smiling, "
          "faded ash gray messy bun hair+, border light, depth of field, large breasts++, looking at "
          "viewer, closeup")

negative_prompt = ("paintings, sketches, worst quality+++++, low quality+++++, normal quality+++++, lowres, "
                   "normal quality, monochrome++, grayscale++, skin spots, acnes, skin blemishes, "
                   "age spot, glans")


def get_controlnet_ref_image_dataurl():
    ref_image = Image.open(r"./data/reference_original.jpg")
    buffered = io.BytesIO()
    ref_image.save(buffered, format="PNG")
    return 'data:image/png;base64,' + base64.b64encode(buffered.getvalue()).decode("utf-8")


def gen_image_sd15_original():
    args = GenImageArgs(
        base_model="runwayml/stable-diffusion-v1-5",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_original.png")


def gen_image_sd15_chilloutmix():
    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_chilloutmix.png")


def gen_image_sd15_lora():
    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        lora=LoraArgs(
            model="korean-doll-likeness-v2-0"
        ),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_lora.png")


def gen_image_sd15_lora_vae():
    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        lora=LoraArgs(
            model="korean-doll-likeness-v2-0"
        ),
        vae="stabilityai/sd-vae-ft-mse",
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_lora_vae.png")


def gen_image_sd15_lora_controlnet():

    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        lora=LoraArgs(
            model="korean-doll-likeness-v2-0"
        ),
        controlnet=ControlnetArgs(
            model="lllyasviel/sd-controlnet-canny",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        ),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_lora_controlnet.png")


def gen_image_sdxl():
    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, steps=40)
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl.png")


def gen_image_sdxl_refiner():
    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9),
        refiner=RefinerArgs(
            model="stabilityai/stable-diffusion-xl-refiner-1.0"
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_refiner.png")


def gen_image_sdxl_controlnet():

    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, steps=40),
        controlnet=ControlnetArgs(
            model="diffusers/controlnet-canny-sdxl-1.0",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_controlnet.png")


def gen_image_sdxl_controlnet_refiner():

    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(
            num_images=9,
            steps=30
        ),
        controlnet=ControlnetArgs(
            model="diffusers/controlnet-canny-sdxl-1.0",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        ),
        refiner=RefinerArgs(
            model="stabilityai/stable-diffusion-xl-refiner-1.0",
            step=30
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_controlnet_refiner.png")


def gen_image_sdxl_lora():
    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt="desert, full shot, dark stillsuit, stillsuit mask up, gloves, solo, highly detailed eyes, "
               "hyper-detailed, high quality visuals, dim Lighting, ultra-realistic, sharply focused, octane render, "
               "8k UHD",
        negative_prompt="no moon++, buried in sand, bare hands, figerless gloves, "
                        "blue stillsuit, barefoot, weapon, vegetation, clouds, glowing eyes++, helmet, "
                        "bare handed, no gloves, double mask, simplified, abstract, unrealistic, impressionistic, "
                        "low resolution,",
        task_config=TaskConfig(num_images=9, steps=40),
        lora=LoraArgs(
            model="stillsuit-sdxl"
        ),
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora.png")


def gen_image_sdxl_lora_refiner():
    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt="desert, full shot, dark stillsuit, stillsuit mask up, gloves, solo, highly detailed eyes, "
               "hyper-detailed, high quality visuals, dim Lighting, ultra-realistic, sharply focused, octane render, "
               "8k UHD",
        negative_prompt="no moon++, buried in sand, bare hands, figerless gloves, "
                        "blue stillsuit, barefoot, weapon, vegetation, clouds, glowing eyes++, helmet, "
                        "bare handed, no gloves, double mask, simplified, abstract, unrealistic, impressionistic, "
                        "low resolution,",
        task_config=TaskConfig(num_images=9, steps=40),
        lora=LoraArgs(
            model="stillsuit-sdxl"
        ),
        refiner=RefinerArgs(
            model="stabilityai/stable-diffusion-xl-refiner-1.0",
            step=30
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora_refiner.png")


def gen_image_sdxl_lora_controlnet():
    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt="desert, full shot, dark stillsuit, stillsuit mask up, gloves, solo, highly detailed eyes, "
               "hyper-detailed, high quality visuals, dim Lighting, ultra-realistic, sharply focused, octane render, "
               "8k UHD",
        negative_prompt="no moon++, buried in sand, bare hands, figerless gloves, "
                        "blue stillsuit, barefoot, weapon, vegetation, clouds, glowing eyes++, helmet, "
                        "bare handed, no gloves, double mask, simplified, abstract, unrealistic, impressionistic, "
                        "low resolution,",
        task_config=TaskConfig(num_images=9, steps=40),
        lora=LoraArgs(
            model="stillsuit-sdxl"
        ),
        controlnet=ControlnetArgs(
            model="diffusers/controlnet-canny-sdxl-1.0",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora_controlnet.png")


def gen_image_sdxl_lora_controlnet_refiner():
    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt="desert, full shot, dark stillsuit, stillsuit mask up, gloves, solo, highly detailed eyes, "
               "hyper-detailed, high quality visuals, dim Lighting, ultra-realistic, sharply focused, octane render, "
               "8k UHD",
        negative_prompt="no moon++, buried in sand, bare hands, figerless gloves, "
                        "blue stillsuit, barefoot, weapon, vegetation, clouds, glowing eyes++, helmet, "
                        "bare handed, no gloves, double mask, simplified, abstract, unrealistic, impressionistic, "
                        "low resolution,",
        task_config=TaskConfig(num_images=9, steps=40),
        lora=LoraArgs(
            model="stillsuit-sdxl"
        ),
        controlnet=ControlnetArgs(
            model="diffusers/controlnet-canny-sdxl-1.0",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        ),
        refiner=RefinerArgs(
            model="stabilityai/stable-diffusion-xl-refiner-1.0",
            step=30
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_lora_controlnet_refiner.png")


def mismatch_sd15_base_lora():
    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        lora=LoraArgs(
            model="gegants_2_1_768"
        ),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_mismatch.png")


def mismatch_sd15_base_controlnet():
    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        controlnet=ControlnetArgs(
            model="diffusers/controlnet-canny-sdxl-1.0",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        ),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_mismatch.png")


def mismatch_sd15_base_vae():
    args = GenImageArgs(
        base_model="emilianJR/chilloutmix_NiPrunedFp32Fix",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        lora=LoraArgs(
            model="gegants_2_1_768"
        ),
        safety_checker=False,
        vae="runwayml/stable-diffusion-v1-5",
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sd15_mismatch.png")


def mismatch_sdxl_base_refiner():
    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9),
        refiner=RefinerArgs(
            model="stabilityai/stable-diffusion-xl-base-1.0"
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_mismatch.png")


def mismatch_sdxl_base_lora():
    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, safety_checker=False),
        lora=LoraArgs(
            model="korean-doll-likeness-v2-0"
        ),
        safety_checker=False
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_mismatch.png")
    pass


def mismatch_sdxl_base_controlnet():
    ref_image_dataurl = get_controlnet_ref_image_dataurl()

    args = GenImageArgs(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        prompt=prompt,
        negative_prompt=negative_prompt,
        task_config=TaskConfig(num_images=9, steps=40),
        controlnet=ControlnetArgs(
            model="lllyasviel/sd-controlnet-canny",
            image_dataurl=ref_image_dataurl,
            preprocess=PreprocessArgs(
                method="canny"
            )
        )
    )

    # Generate images
    images = gen_images(args)

    # Save the images in a grid
    image_grid = make_image_grid(images, 3, 3)
    image_grid.save("./data/sdxl_mismatch.png")


if __name__ == "__main__":
    load_config()
    gen_image_sd15_original()
    gen_image_sd15_chilloutmix()
    # gen_image_sd15_lora()
    # gen_image_sd15_lora_vae()
    # gen_image_sd15_lora_controlnet()
    gen_image_sdxl()
    gen_image_sdxl_refiner()
    gen_image_sdxl_controlnet()
    gen_image_sdxl_controlnet_refiner()
    # gen_image_sdxl_lora()
    # gen_image_sdxl_lora_refiner()
    # gen_image_sdxl_lora_controlnet()
    # gen_image_sdxl_lora_controlnet_refiner()
    # mismatch_sd15_base_lora()
    # mismatch_sd15_base_controlnet()
    # mismatch_sd15_base_vae()
    # mismatch_sdxl_base_refiner()
    # mismatch_sdxl_base_lora()
    # mismatch_sdxl_base_controlnet()
