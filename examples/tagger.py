from sd_task.task_args.tagger import TaggerTaskArgs
from sd_task.task_runner.tagger import run_tagger_task
from reference_image import get_controlnet_ref_image_dataurl


if __name__ == "__main__":
    ref_image = get_controlnet_ref_image_dataurl()

    args = TaggerTaskArgs(image_dataurl=ref_image, model="WD14 moat tagger v2")

    ratings, tags, discarded_tags = run_tagger_task(args)
    print(tags)
    print(discarded_tags)
