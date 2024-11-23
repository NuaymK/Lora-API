from pydantic import BaseModel
from google.cloud import storage




class TrainRequest(BaseModel):
    dataset_url: str
    output_directory: str
    training_steps: int
    model_name: str
    model_path: str = "/runpod-volume/trained_models"
    resolution: str = "1024,1024"
    instance_prompt: str
    class_prompt: str 



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