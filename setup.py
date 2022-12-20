from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="test-rec-system",
    version="1.0.0",
    packages=find_packages(include=["rec-system"]),
    description="test package",
    # long_description=long_description,
    url="https://github.com/infox182/ml-rec-systems",
    author="Ilya Skryagin",
)
