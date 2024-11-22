from fastapi import FastAPI, HTTPException
import os
import subprocess
import base64
from core.helper import TrainRequest,LoraHelper

app = FastAPI()

GCS_CREDENTIALS_BASE64 = os.getenv("GCS_CREDENTIALS")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

if not GCS_CREDENTIALS_BASE64:
    raise ValueError("GCS_CREDENTIALS environment variable is not set.")

GCS_CREDENTIALS_PATH = "/app/gcs-key.json"
with open(GCS_CREDENTIALS_PATH, "w") as f:
    f.write(base64.b64decode(GCS_CREDENTIALS_BASE64).decode("utf-8"))

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_PATH





@app.post("/train")
def train_model(request: TrainRequest):
    try:
        dataset_url = request.dataset_url
        output_dir = request.output_directory
        training_steps = request.training_steps
        pretrained_model_path = "/runpod-volume/base_model/sdxl_base_model.safetensors"  
        model_name = request.model_name
        model_path = request.model_path
        resolution = request.resolution

        os.makedirs(output_dir, exist_ok=True)

        dataset_zip = os.path.join(output_dir, "dataset.zip")
        subprocess.run(["wget", dataset_url, "-O", dataset_zip], check=True)
        subprocess.run(["unzip", dataset_zip, "-d", output_dir], check=True)

        LoraHelper.restructure_dataset(output_dir,model_name)


        command = [
            "accelerate", "launch", "--num_cpu_threads_per_process", "1", "/app/kohya_ss/sdxl_train_network.py",
            "--pretrained_model_name_or_path", pretrained_model_path,
            "--train_data_dir", output_dir,
            "--output_dir", model_path,
            "--max_train_steps", str(training_steps),
            "--output_name", model_name,
            "--resolution", resolution,
            "--save_model_as", "safetensors",
            "--network_module", "networks.lora",
            "--train_batch_size", "1",  
            "--gradient_checkpointing",  
            "--mixed_precision", "fp16",
            "--max_token_length", "150",
            "--face_crop_aug_range", "0.5,1.0",
            "--random_crop",
            "--bucket_reso_steps", "64",
        ]
        subprocess.run(command, check=True)

        trained_model_path = os.path.join(model_path, f"{model_name}.safetensors")
        LoraHelper.upload_to_gcs(trained_model_path, f"trained_models/{model_name}.safetensors",BUCKET_NAME)

        return {"message": "Training completed successfully.", "model_url": f"trained_models/{model_name}.safetensors"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
