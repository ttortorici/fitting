from setuptools import setup, find_packages

setup(
    name='fitting',
    version='1.0',
    description='Least squares fitting of data to various models for my PhD thesis',
    author='Teddy Tortorici',
    author_email='edward.tortorici@colorado.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ],
)