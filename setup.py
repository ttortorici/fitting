from setuptools import setup, find_packages

setup(
    name='fitting',
    version='1.1',
    description='Least squares fitting of data to various models for my PhD thesis',
    author='Teddy Tortorici',
    author_email='edward.tortorici@colorado.edu',
    packages=find_packages(include=['fitting', 'fitting.*']),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'fabio'
    ],
)