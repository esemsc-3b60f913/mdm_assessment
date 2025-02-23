# setup.py
from setuptools import setup, find_packages

setup(
    name="acsefunctions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.19"],
    author="Your Name",
    description="A numerical package for computing transcendental functions via Taylor series.",
    python_requires=">=3.8",
)
