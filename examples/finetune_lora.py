from sd_task.task_runner import run_finetune_lora_task
from sd_task.task_args import FinetuneLoraTaskArgs


if __name__ == "__main__":
    args = {
        "model": {
            "name": "runwayml/stable-diffusion-v1-5"
        },
        "dataset": {
            "name": "lambdalabs/naruto-blip-captions",
        },
        "validation": {
            "num_images": 4,
        },
        "train_args": {
            "learning_rate": 1e-4,
            "batch_size": 1,
            "num_train_steps": 100,
            "max_train_steps": 15000,
            "lr_scheduler": {
                "lr_scheduler": "cosine",
                "lr_warmup_steps": 0,
            }
        },
        "lora": {
            "rank": 4,
            "init_lora_weights": "gaussian",
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"]
        },
        "transforms": {
            "center_crop": True,
            "random_flip": True,
        },
        "mixed_precision": "fp16",
        "seed": 1337
    }

    run_finetune_lora_task(FinetuneLoraTaskArgs.model_validate(args), "data/finetune_lora")
