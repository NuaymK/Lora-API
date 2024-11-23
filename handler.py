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
        model_dir = model_path
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
            "--train_data_dir", img_subdir,
            "--output_dir", model_dir,
            "--max_train_steps", str(training_steps),
            "--output_name", model_name,
            "--resolution", resolution,
            "--save_model_as", "safetensors",
            "--network_module", "networks.lora",
            "--train_batch_size", "1",  
            "--gradient_checkpointing",  
            "--mixed_precision", "bf16",
            "--max_token_length", "75",
            "--bucket_reso_steps", "64",
            "--enable_bucket",
            "--cache_latents_to_disk",
            "--optimizer", "Adafactor",
            "--optimizer_args", "scale_parameter=False relative_step=False warmup_init=False",
            "--clip_skip", "1",
            "--network_dim", "256",
            "--network_alpha", "1",
            "--unet_lr", "0.0003",
            "--text_encoder_lr", "0.0003",
            "--lr_scheduler", "constant",
            "--max_bucket_reso", "2048",
            "--min_bucket_reso", "256",
            "--save_precision", "bf16",
            "--sdxl_no_half_vae",
            "--xformers",
            "--LoRA_type", "Standard",
            "--adaptive_noise_scale", "0",
            "--additional_parameters", "",
            "--block_alphas", "",
            "--block_dims", "",
            "--block_lr_zero_threshold", "",
            "--bucket_no_upscale",
            "--cache_latents",
            "--caption_dropout_every_n_epochs", "0.0",
            "--caption_dropout_rate", "0",
            "--caption_extension", ".txt",
            "--color_aug", "false",
            "--conv_alpha", "1",
            "--conv_block_alphas", "",
            "--conv_block_dims", "",
            "--conv_dim", "1",
            "--decompose_both", "false",
            "--dim_from_weights", "false",
            "--down_lr_weight", "",
            "--epoch", "10",
            "--factor", "-1",
            "--flip_aug", "false",
            "--full_bf16", "false",
            "--full_fp16", "false",
            "--gradient_accumulation_steps", "1.0",
            "--keep_tokens", "0",
            "--lora_network_weights", "",
            "--lr_scheduler_num_cycles", "",
            "--lr_scheduler_power", "",
            "--lr_warmup", "0",
            "--max_data_loader_n_workers", "0",
            "--max_timestep", "1000",
            "--max_train_epochs", "",
            "--mem_eff_attn", "false",
            "--mid_lr_weight", "",
            "--min_snr_gamma", "0",
            "--min_timestep", "0",
            "--model_list", "custom",
            "--module_dropout", "0",
            "--multires_noise_discount", "0",
            "--multires_noise_iterations", "0",
            "--network_dropout", "0",
            "--no_token_padding", "false",
            "--noise_offset", "0",
            "--noise_offset_type", "Original",
            "--persistent_data_loader_workers", "false",
            "--prior_loss_weight", "1.0",
            "--random_crop", "false",
            "--rank_dropout", "0",
            "--resume", "",
            "--sample_every_n_epochs", "0",
            "--sample_every_n_steps", "0",
            "--sample_prompts", "",
            "--sample_sampler", "euler_a",
            "--save_every_n_epochs", "1",
            "--save_every_n_steps", "0",
            "--save_last_n_steps", "0",
            "--save_last_n_steps_state", "0",
            "--save_state", "false",
            "--scale_v_pred_loss_like_noise_pred", "false",
            "--scale_weight_norms", "0",
            "--sdxl", "true",
            "--sdxl_cache_text_encoder_outputs", "false",
            "--seed", "",
            "--shuffle_caption", "false",
            "--stop_text_encoder_training", "0",
            "--train_on_input", "true",
            "--training_comment", "",
            "--unit", "1",
            "--up_lr_weight", "",
            "--use_cp", "false",
            "--use_wandb", "false",
            "--v2", "false",
            "--v_parameterization", "false",
            "--vae_batch_size", "0",
            "--wandb_api_key", "",
            "--weighted_captions", "false"
        ]
        subprocess.run(command, check=True)

        trained_model_path = os.path.join(model_dir, f"{model_name}.safetensors")
        LoraHelper.upload_to_gcs(trained_model_path, f"trained_models/{model_name}.safetensors", BUCKET_NAME)

        return {"message": "Training completed successfully.", "model_url": f"trained_models/{model_name}.safetensors"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

