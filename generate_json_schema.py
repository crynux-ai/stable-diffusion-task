from sd_task.task_args import InferenceTaskArgs, FinetuneLoraTaskArgs
import json

if __name__ == '__main__':
    dest_file = "./schema/stable-diffusion-inference-task.json"

    schema = InferenceTaskArgs.model_json_schema()

    with open(dest_file, "w") as f:
        json.dump(schema, f)

    dest_file = "./schema/stable-diffusion-finetune-lora-task.json"

    schema = FinetuneLoraTaskArgs.model_json_schema()

    with open(dest_file, mode="w") as f:
        json.dump(schema, f)

    print("json schema output successfully")
