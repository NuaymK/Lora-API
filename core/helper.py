from core import worker_config
from b2sdk.api import B2Api
from b2sdk.account_info import InMemoryAccountInfo


class LoraHelper:
    # Hardcoded Backblaze credentials (ensure these are secured in production)
    ACCOUNT_ID = '005009c8de9b4750000000001'  # Your Account ID
    APPLICATION_KEY = 'K005oQ5rhKH1SZg00mKv3WbDbwbIIyE'  # Your Application Key
    BUCKET_NAME = 'LoRA-Models-VVG'  # Your bucket name

    @staticmethod
    def initialize_backblaze():
        """
        Initializes Backblaze API authorization and returns a B2Api instance.
        """
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", LoraHelper.ACCOUNT_ID,
                               LoraHelper.APPLICATION_KEY)
        return b2_api

    @staticmethod
    def upload_to_backblaze(local_file_path, destination_blob_name):
        """
        Uploads a file to Backblaze B2 and returns the file's download URL.
        """
        try:
            # Initialize the Backblaze API
            b2_api = LoraHelper.initialize_backblaze()

            # Get the target bucket by name
            bucket = b2_api.get_bucket_by_name(LoraHelper.BUCKET_NAME)

            # Read the file and upload it to Backblaze
            with open(local_file_path, 'rb') as file_object:
                file_data = file_object.read()
                file_info = bucket.upload_bytes(file_data, destination_blob_name)

            # Retrieve the download base URL from the account info
            download_base_url = b2_api.account_info.get_download_url()

            # Construct the public download URL
            download_url = (
                f"{download_base_url}/file/"
                f"{LoraHelper.BUCKET_NAME}/{file_info.file_name}"
            )

            print(
                f"Uploaded {local_file_path} to Backblaze.\nDownload URL: {download_url}"
            )
            return download_url

        except Exception as e:
            print(f"Failed to upload {local_file_path} to Backblaze: {e}")
            raise

    @staticmethod
    def get_schema():
        """
        Returns the input schema from worker_config.
        """
        return worker_config.INPUT_SCHEMA

