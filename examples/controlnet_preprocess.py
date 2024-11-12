from sd_task.task_args.controlnet_preprocess import ControlnetPreprocessTaskArgs, PreprocessMethodLineartRealistic
from sd_task.task_runner.controlnet_preprocess import run_controlnet_preprocess_task
from reference_image import get_controlnet_ref_image_dataurl

if __name__ == "__main__":
    ref_image = get_controlnet_ref_image_dataurl()

    args = ControlnetPreprocessTaskArgs(
        image_dataurl=ref_image,
        preprocess=PreprocessMethodLineartRealistic()
    )
    image = run_controlnet_preprocess_task(args)
    image.save("./data/controlnet_preprocess.png")
