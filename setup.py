from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

setup(
    name="dataset_manipulator",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[
        "fire",
        "tabulate",
        "pyyaml",
        "imagehash",
        "matplotlib",
        "opencv-python",
    ],
    entry_points={
        "console_scripts": [
            "dsforge=app.cli:main",
        ],
    },
)