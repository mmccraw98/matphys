from setuptools import setup, find_packages
# import sys
# import os

# sys.path.append('/home/mmccraw/dev')
# os.environ['PYTHONPATH'] = '/path/to/your/pythonpath'

# Parse requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="matphys",
    version="0.1",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)