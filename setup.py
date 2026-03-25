from setuptools import setup, find_packages


setup(
    name="dataset_manipulator",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "fire",
    ],
    entry_points={
        "console_scripts": [
            "app=app.cli:main",
        ],
    },
)