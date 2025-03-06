import os
import subprocess
import base64
from core.helper import LoraHelper
import runpod
from runpod.serverless.utils.rp_validator import validate


def train_model(data):
    try:
        data_input = data['input']

        if 'errors' in (data_input := validate(data_input, LoraHelper.get_schema())):
            return {'error': data_input['errors']}
        
        job_input = data_input['validated_input']

        dataset_url = job_input["dataset_url"]
        output_dir = job_input["output_directory"]
        training_steps = job_input["training_steps"]
        pretrained_model_path = "/runpod-volume/base_model/flux1-dev2pro.safetensors"  
        model_name = job_input["model_name"]
        model_path = job_input["model_path"]
        resolution = "1024,1024"
        instance_prompt = job_input["instance_prompt"]  
        class_prompt = job_input["class_prompt"]

        out_dir = os.path.join(output_dir, "out")
        img_dir = os.path.join(out_dir, "img")
        log_dir = os.path.join(out_dir, "log")
        img_subdir = os.path.join(img_dir, f"20_{instance_prompt} {class_prompt}")

        os.makedirs(img_subdir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        dataset_zip = os.path.join(output_dir, "dataset.zip")
        temp_extract_dir = os.path.join(output_dir, "temp_extract")
        
        # Create temp directory for initial extraction
        os.makedirs(temp_extract_dir, exist_ok=True)
        
        # Download the dataset
        subprocess.run(["wget", dataset_url, "-O", dataset_zip], check=True)
        
        # Extract to temporary directory
        subprocess.run(f"unzip '{dataset_zip}' -d '{temp_extract_dir}'", shell=True, check=True)
        
        # Find all image files recursively and move them to the target directory
        image_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')
        
        # Find all image files recursively
        find_cmd = f"find '{temp_extract_dir}' -type f -iregex '.*\\.\\(jpg\\|jpeg\\|png\\|webp\\|bmp\\|tiff\\|tif\\)'"
        image_files = subprocess.run(find_cmd, shell=True, check=True, capture_output=True, text=True).stdout.strip().split('\n')
        
        # Copy each image file to the target directory
        for img_file in image_files:
            if img_file:  # Skip empty lines
                # Get just the filename without path
                filename = os.path.basename(img_file)
                # Copy the file to target directory
                subprocess.run(f"cp '{img_file}' '{img_subdir}/{filename}'", shell=True, check=True)
        
        # Clean up temporary directory
        subprocess.run(f"rm -rf '{temp_extract_dir}'", shell=True, check=True)

        # Generate dataset.toml
        dataset_toml_path = os.path.join(output_dir, "dataset.toml")
        with open(dataset_toml_path, "w") as f:
            f.write(f"""[general]
shuffle_caption = false
caption_extension = '.txt'
keep_tokens = 1

[[datasets]]
resolution = 1024
batch_size = 1
keep_tokens = 1

  [[datasets.subsets]]
  image_dir = '{img_subdir}'
  class_tokens = '{instance_prompt} {class_prompt}'
  num_repeats = 1
            """)

        command = [
            "accelerate", "launch", 
            "--mixed_precision", "bf16",
            "--num_cpu_threads_per_process", "1",
            "/app/kohya_ss/flux_train_network.py",
            "--pretrained_model_name_or_path", pretrained_model_path,
            "--clip_l", "/runpod-volume/base_model/clip_l.safetensors",
            "--t5xxl", "/runpod-volume/base_model/t5xxl_fp16.safetensors",
            "--ae", "/runpod-volume/base_model/ae.safetensors",
            "--cache_latents_to_disk",
            "--save_model_as", "safetensors",
            "--sdpa",
            "--persistent_data_loader_workers",
            "--max_data_loader_n_workers", "2",
            "--seed", "42",
            "--gradient_checkpointing",
            "--mixed_precision", "bf16",
            "--save_precision", "bf16",
            "--network_module", "networks.lora_flux",
            "--network_dim", "4",
            "--optimizer_type", "adamw8bit",
            "--learning_rate", "0.0008",
            "--cache_text_encoder_outputs",
            "--cache_text_encoder_outputs_to_disk",
            "--fp8_base",
            "--highvram",
            "--max_train_epochs", "1",
            "--dataset_config", dataset_toml_path,
            "--output_dir", model_path,
            "--output_name", model_name,
            "--timestep_sampling", "shift",
            "--discrete_flow_shift", "3.1582",
            "--model_prediction_type", "raw",
            "--guidance_scale", "1",
            "--loss_type", "l2"
        ]
        subprocess.run(command, check=True)

        trained_model_path = os.path.join(model_path, f"{model_name}.safetensors")
        download_url = LoraHelper.upload_to_backblaze(trained_model_path, f"trained_models/{model_name}.safetensors")

        return {"message": "Training completed successfully.", "model_url": download_url}
    except Exception as e:
        print(f"Failed to train the model: {e}")
        raise

runpod.serverless.start({"handler": train_model})
