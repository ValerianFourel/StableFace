from setuptools import setup, find_packages

setup(
    name='StableFace',
    version='0.1.0',
    description='A tool for analyzing and improving motion stability in talking face generation',
    author='Valerian Fourel',
    author_email='valerian.fourel@gmail.com',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # List your dependencies here
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        # Add any other dependencies
    ],
    python_requires='>=3.6',
    include_package_data=True,  # To include non-python files like .yml
    package_data={
        '': ['*.yml'],  # Include .yml files in the package
    },
)
