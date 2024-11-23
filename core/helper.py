from core import worker_config
from google.cloud import storage






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
    def get_schema():
        return worker_config.INPUT_SCHEMA