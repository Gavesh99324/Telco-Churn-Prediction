# Package setup
from setuptools import setup, find_packages

setup(
    name="telco-churn-prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
    ],
)
