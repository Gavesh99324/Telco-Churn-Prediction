"""Spark session"""
from pyspark.sql import SparkSession


class SparkSessionManager:
    """Manage Spark session"""

    @staticmethod
    def get_spark_session(app_name='ChurnPrediction'):
        """Get or create Spark session"""
        return SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
