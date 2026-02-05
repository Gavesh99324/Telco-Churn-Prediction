"""Spark utilities"""


class SparkUtils:
    """Spark utility functions"""

    @staticmethod
    def read_csv(spark, file_path, header=True, inferSchema=True):
        """Read CSV file using Spark"""
        return spark.read.csv(file_path, header=header, inferSchema=inferSchema)
