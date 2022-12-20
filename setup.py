from setuptools import setup, find_packages

setup(
    name='torch-bce',
    version='0.1.0',
    packages=find_packages(include=['torch_bce', 'torch_bce.*'],
                           exclude=['torch_bce.tests']
                           ),
    install_requires=[
        "numpy>=1.23.0",
        "pandas>=1.3.5",
        "torch>=1.12.0",
        "wandb>=0.13.3",
        "torchmetrics>=0.10.1"
    ],
    author="Dmitrii Trofimov",
    author_email="dmitry.trofimow2011@gmail.com"
)