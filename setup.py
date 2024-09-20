#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.md") as history_file:
    history = history_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = ['pytest', 'flake8', 'coverage']

setup(
    author="Anupama Jha",
    author_email="anupamaj@uw.edu",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Predicting trans 3D genome folding from DNA sequences using TwinC.",
    entry_points={
        "console_scripts": [
            "twinc_train=twinc.twinc_train:main",
            "twinc_test=twinc.twinc_test:main",
        ],
    },
    install_requires=requirements,
    license="Apache license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="twinc",
    name="twinc",
    packages=["twinc"],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Noble-Lab/twinc",
    version="0.1.0",
    zip_safe=False,
)