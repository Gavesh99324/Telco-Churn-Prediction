"""S3 artifact management"""
from utils.s3_io import S3Handler


class S3ArtifactManager:
    """Manage ML artifacts in S3"""

    def __init__(self, bucket_name):
        self.s3_handler = S3Handler(bucket_name)

    def save_artifact(self, artifact_path, s3_key):
        """Save artifact to S3"""
        self.s3_handler.upload_file(artifact_path, s3_key)

    def load_artifact(self, s3_key, local_path):
        """Load artifact from S3"""
        self.s3_handler.download_file(s3_key, local_path)
