from setuptools import find_packages, setup 

setup(
    name='cbas_headless',
    packages=find_packages(),
    package_data={'cbas_headless.platforms.modified_deepethogram.conf': ['**/*.yaml'], 'cbas_headless.assets': ['*.png','*.gif']},
    version='0.1.0',
    description='The Circadian Behavioral Analysis Suite',
    author='Logan Pery',
    install_requires=[],
)