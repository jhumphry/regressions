from setuptools import setup, find_packages

setup(
    name='Regressions',
    version='0.1.0',

    packages=find_packages(),
    install_requires=['numpy>=1.10'],

    description='Implementations of various regression algorithms, including '
                'Partial Least Squares and Principal Components Regression',
    url='https://github.com/jhumphry/regressions',
    license='ISC',

    zip_safe=True
)
