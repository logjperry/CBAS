from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'CBAS: A Circadian Behavioral Analysis Suite'
LONG_DESCRIPTION = 'A toolkit for recording and analyzing animal behavior on a circadian timescale.'

# Setting up
setup(
        name="cbas_headless", 
        version=VERSION,
        author="Logan Perry",
        author_email="<loganperry@tamu.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], 
        
        keywords=['python', 'cbas'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: Microsoft :: Windows"
        ]
)