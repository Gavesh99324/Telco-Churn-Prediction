"""Test data ingestion"""
import pytest
from src.data_ingestion import DataIngestion


def test_data_ingestion():
    """Test data ingestion module"""
    ingestion = DataIngestion()
    assert ingestion is not None
