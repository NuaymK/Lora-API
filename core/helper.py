import subprocess
import json
from core import worker_config


class LoraHelper:
    # Hardcoded Backblaze credentials (not recommended for production)
    ACCOUNT_ID = '005009c8de9b4750000000001'
    APPLICATION_KEY = 'K005oQ5rhKH1SZg00mKv3WbDbwbIIyE'
    BUCKET_NAME = 'LoRA-Models-VVG'
    # Adjust this hostname to match your account’s download domain (if needed)
    DOWNLOAD_HOST = 'f005.backblazeb2.com'

    @staticmethod
    def run_command(command):
        """
        Runs a shell command and returns its stdout.
        Raises subprocess.CalledProcessError if the command fails.
        """
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        return result.stdout.strip()

    @staticmethod
    def authorize():
        """
        Authorizes the B2 CLI with your Backblaze credentials.
        This command sets up the CLI for subsequent operations.
        """
        try:
            command = [
                "b2",
                "authorize_account",
                LoraHelper.ACCOUNT_ID,
                LoraHelper.APPLICATION_KEY
            ]
            print("Running authorization command:", " ".join(command))
            output = LoraHelper.run_command(command)
            print("Authorization output:", output)
            return output
        except subprocess.CalledProcessError as e:
            print("Failed to authorize Backblaze account.")
            print("Error:", e.stderr)
            raise

    @staticmethod
    def upload_to_backblaze(local_file_path, destination_blob_name):
        """
        Uploads a file to Backblaze B2 using the B2 CLI.

        Steps:
          1. Authorize with Backblaze using hardcoded credentials.
          2. Run the 'b2 upload_file' command.
          3. Parse output to determine the file name.
          4. Construct and return a download URL.

        Parameters:
          - local_file_path: Path to the local file.
          - destination_blob_name: The file name to assign in B2.
          
        Returns:
          The constructed download URL for the uploaded file.
        """
        try:
            # First, authorize the B2 CLI
            LoraHelper.authorize()

            # Build the upload command
            command = [
                "b2",
                "upload_file",
                "--bucket", LoraHelper.BUCKET_NAME,
                local_file_path,
                destination_blob_name
            ]
            print("Running upload command:", " ".join(command))
            output = LoraHelper.run_command(command)
            print("Upload output:", output)

            # Attempt to parse the upload output as JSON
            try:
                file_info = json.loads(output)
                file_name = file_info.get("fileName", destination_blob_name)
            except json.JSONDecodeError:
                # Fallback when output is not in JSON format
                file_name = destination_blob_name

            # Construct the download URL; adjust the host if needed
            download_url = (
                f"https://{LoraHelper.DOWNLOAD_HOST}/file/"
                f"{LoraHelper.BUCKET_NAME}/{file_name}"
            )
            print(
                f"Uploaded {local_file_path} to Backblaze. "
                f"Download URL: {download_url}"
            )
            return download_url

        except subprocess.CalledProcessError as e:
            print("Failed to upload to Backblaze. Error:", e.stderr)
            raise

    @staticmethod
    def get_schema():
        """
        Returns the input schema from your worker configuration.
        """
        return worker_config.INPUT_SCHEMA
