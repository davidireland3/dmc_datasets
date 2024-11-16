from setuptools import setup, find_packages

setup(
    name="dmc_datasets",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "dm_control",
        "gymnasium"
    ],
    author="David Ireland, Alex Beeson",
    author_email="alex.beeson@warwick.ac.uk",
    description="Discrete action space wrapper for DeepMind Control Suite and wrapper"
                " for D4RL formatted DMC datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
