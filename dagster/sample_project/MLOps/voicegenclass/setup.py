from setuptools import find_packages, setup

setup(
    name="voicegenclass",
    packages=find_packages(exclude=["voicegenclass_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
