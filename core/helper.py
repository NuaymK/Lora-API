from b2sdk.v1 import B2Api, InMemoryAccountInfo

class LoraHelper():
    # Hardcoded Backblaze credentials
    ACCOUNT_ID = '005009c8de9b4750000000001'  # Your Account ID
    APPLICATION_KEY = 'K005oQ5rhKH1SZg00mKv3WbDbwbIIyE'  # Your Application Key
    BUCKET_NAME = 'LoRA-Models-VVG'  # Replace with your bucket name
    
    @staticmethod
    def initialize_backblaze():
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", LoraHelper.ACCOUNT_ID, LoraHelper.APPLICATION_KEY)
        return b2_api

    @staticmethod
    def upload_to_backblaze(local_file_path, destination_blob_name):
        try:
            b2_api = LoraHelper.initialize_backblaze()
            bucket = b2_api.get_bucket_by_name(LoraHelper.BUCKET_NAME)
            
            with open(local_file_path, 'rb') as file_data:
                file_info = bucket.upload_bytes(
                    file_data.read(), 
                    destination_blob_name
                )
            
            # Get the download URL
            download_url = f"https://{LoraHelper.BUCKET_NAME}.backblazeb2.com/file/{file_info.file_name}"
            print(f"Uploaded {local_file_path} to Backblaze. Download URL: {download_url}")
            return download_url
            
        except Exception as e:
            print(f"Failed to upload to Backblaze: {e}")
            raise

    @staticmethod
    def get_schema():
        return worker_config.INPUT_SCHEMA
