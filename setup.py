from __future__ import annotations

from setuptools import find_packages
from setuptools import setup

setup(
    name="dataset_manipulator",
    version="0.0.2",
    # python_requires="=>3.11",
    packages=find_packages(),
    install_requires=[

    ],
    entry_points={
        "console_scripts": [
            "app=app.cli:main",
        ],
    },
)
