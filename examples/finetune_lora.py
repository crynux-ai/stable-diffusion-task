import os
import shutil

from sd_task.task_args import FinetuneLoraTaskArgs
from sd_task.task_runner import run_finetune_lora_task

if __name__ == "__main__":
    output_dir = "data/finetune_lora"
    checkpoint_dir = None
    i = 0
    while True:
        args = {
            "model": {
                "name": "crynux-ai/stable-diffusion-v1-5",
                "variant": "fp16"
            },
            "dataset": {
                "url": "https://gateway.irys.xyz/GivF5FBMdJVr6xHT7hi2aE7vH55wVjrtKLpRc2E86icJ"
            },
            "validation": {
                "num_images": 4,
            },
            "train_args": {
                "learning_rate": 1e-4,
                "batch_size": 1,
                "num_train_steps": 100,
                "max_train_steps": 200,
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
            "seed": 1337,
            "checkpoint": checkpoint_dir
        }

        run_finetune_lora_task(FinetuneLoraTaskArgs.model_validate(args), output_dir=output_dir)
        origin_checkpoint_dir = os.path.join(output_dir, "checkpoint")

        if os.path.exists(os.path.join(origin_checkpoint_dir, "FINISH")):
            result_dir = os.path.join(output_dir, "result")
            shutil.move(origin_checkpoint_dir, result_dir)
            print(f"Training finished, result saved to {result_dir}")
            break

        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{i + 1}")
        shutil.move(origin_checkpoint_dir, checkpoint_dir)
        print(f"Checkpoint saved to {checkpoint_dir}")

        i += 1
