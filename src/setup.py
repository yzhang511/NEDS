from setuptools import setup, find_packages

setup(
    name='neds',
    version='0.0.1',
    description='Neural Encoding and Decoding at Scale',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'timm'
    ],
)
