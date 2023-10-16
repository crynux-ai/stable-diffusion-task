from sd_task.inference_task_args.task_args import InferenceTaskArgs
import json

if __name__ == '__main__':
    dest_file = "./schema/stable-diffusion-inference-task.json"

    json_schema_dict = InferenceTaskArgs.model_json_schema()
    json_schema_str = json.dumps(json_schema_dict)

    with (open(dest_file, "wb") as f):
        f.write(bytes(json_schema_str, "utf-8"))

    f.close()
    print("json schema output successfully")
