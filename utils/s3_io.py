"""S3 operations"""
import boto3


class S3Handler:
    """Handle S3 operations"""

    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name

    def upload_file(self, file_path, s3_key):
        """Upload file to S3"""
        self.s3.upload_file(file_path, self.bucket_name, s3_key)

    def download_file(self, s3_key, file_path):
        """Download file from S3"""
        self.s3.download_file(self.bucket_name, s3_key, file_path)
