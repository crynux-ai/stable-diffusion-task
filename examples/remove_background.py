from sd_task.task_args.remove_background import RemoveBackgroundTaskArgs
from sd_task.task_runner.remove_background import run_remove_background_task
from reference_image import get_controlnet_ref_image_dataurl


if __name__ == "__main__":
    ref_image = get_controlnet_ref_image_dataurl()
    args = RemoveBackgroundTaskArgs(
        image_dataurl=ref_image,
        model="u2net",
    )
    image = run_remove_background_task(args)
    image.save("./data/remove_bg.png")
