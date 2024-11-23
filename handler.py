from fastapi import FastAPI, HTTPException
import os
import subprocess
import base64
from core.helper import TrainRequest, LoraHelper

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
        instance_prompt = request.instance_prompt  
        class_prompt = request.class_prompt

        out_dir = os.path.join(output_dir, "out")
        img_dir = os.path.join(out_dir, "img")
        log_dir = os.path.join(out_dir, "log")
        img_subdir = os.path.join(img_dir, f"20_{instance_prompt} {class_prompt}")

        os.makedirs(img_subdir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        dataset_zip = os.path.join(output_dir, "dataset.zip")
        subprocess.run(["wget", dataset_url, "-O", dataset_zip], check=True)
        subprocess.run(["unzip", dataset_zip, "-d", img_subdir], check=True)

        command = [
            "accelerate", "launch", "--num_cpu_threads_per_process", "2", "/app/kohya_ss/sdxl_train_network.py",
            "--pretrained_model_name_or_path", pretrained_model_path,
            "--train_data_dir", img_dir,
            "--output_dir", model_path,
            "--max_train_steps", str(training_steps),
            "--output_name", model_name,
            "--resolution", resolution,
            "--save_model_as", "safetensors",
            "--network_module", "networks.lora",
            "--train_batch_size", "1",  
            "--gradient_checkpointing",  
            "--mixed_precision", "bf16",
            "--max_token_length", "150",
            "--bucket_reso_steps", "64",
            "--enable_bucket",
            "--cache_latents_to_disk",
            "--optimizer_type", "adafactor",
            "--optimizer_args", "scale_parameter=False",
            "--optimizer_args", "relative_step=False",
            "--optimizer_args", "warmup_init=False",            
            "--lr_warmup_steps", "100",
            "--learning_rate", "4e-7",
            "--clip_skip", "1",
            "--network_dim", "256",
            "--network_alpha", "1",
            "--unet_lr", "0.0003",
            "--text_encoder_lr", "0.0003",
            "--lr_scheduler", "constant_with_warmup",
            "--max_bucket_reso", "2048",
            "--min_bucket_reso", "256",
            "--save_precision", "bf16",
            "--xformers",
            "--adaptive_noise_scale", "0",
            "--bucket_no_upscale",
            "--cache_latents",
            "--caption_dropout_every_n_epochs", "0",
            "--caption_dropout_rate", "0",
            "--caption_extension", ".txt",
            "--gradient_accumulation_steps", "1",
            "--keep_tokens", "0",
            "--lr_warmup", "0",
            "--max_data_loader_n_workers", "0",
            "--max_timestep", "1000",
            "--min_snr_gamma", "0",
            "--min_timestep", "0",
            "--multires_noise_discount", "0",
            "--multires_noise_iterations", "0",
            "--network_dropout", "0",
            "--noise_offset", "0",
            "--prior_loss_weight", "1",
            "--resume", "",
            "--sample_every_n_epochs", "0",
            "--sample_every_n_steps", "0",
            "--sample_prompts", "",
            "--sample_sampler", "euler_a",
        ]
        subprocess.run(command, check=True)

        trained_model_path = os.path.join(model_path, f"{model_name}.safetensors")
        LoraHelper.upload_to_gcs(trained_model_path, f"trained_models/{model_name}.safetensors", BUCKET_NAME)

        return {"message": "Training completed successfully.", "model_url": f"trained_models/{model_name}.safetensors"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

