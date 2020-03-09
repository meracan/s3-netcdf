import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meracan-s3netcdf",
    version="0.0.1",
    license='MIT',
    author="Julien Cousineau",
    author_email="Julien.Cousineau@gmail.com",
    description="Create partition netcdf files on s3",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/meracan/s3-netcdf",
    packages=["s3netcdf"],
    install_requires=['numpy','netcdf4','boto3'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)