from diffusers import schedulers


def add_scheduler_pipeline_args(pipeline, args):
    args_dict = {}

    if (hasattr(args, 'args')
            and args.args is not None):
        args_dict = args.args.model_dump(exclude_none=True)

    scheduler = getattr(schedulers, args.method)

    pipeline.scheduler = scheduler.from_config(
        pipeline.scheduler.config,
        **args_dict
    )
