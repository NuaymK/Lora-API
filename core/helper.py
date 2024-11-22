from pydantic import BaseModel
import shutil
import os
from google.cloud import storage
import shutil




class TrainRequest(BaseModel):
    dataset_url: str
    output_directory: str
    training_steps: int
    model_name: str
    model_path: str = "/runpod-volume/trained_models"
    resolution: str = "1024,1024"



class LoraHelper():

    @staticmethod
    def upload_to_gcs(local_file_path, destination_blob_name,BUCKET_NAME):
        try:
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(local_file_path)
            print(f"Uploaded {local_file_path} to {destination_blob_name}.")
        except Exception as e:
            print(f"Failed to upload to GCS: {e}")
            raise


    @staticmethod
    def restructure_dataset(train_data_dir,model_name):
        if not any(os.path.isdir(os.path.join(train_data_dir, d)) for d in os.listdir(train_data_dir)):
            class_dir = os.path.join(train_data_dir, f"5_output {model_name}")
            os.makedirs(class_dir, exist_ok=True)

            for file in os.listdir(train_data_dir):
                file_path = os.path.join(train_data_dir, file)
                if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.move(file_path, class_dir)


    @staticmethod
    def cleanup_directory(directory_path):
        for root, dirs, files in os.walk(directory_path, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))