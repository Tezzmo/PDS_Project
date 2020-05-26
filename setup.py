from setuptools import setup

setup(
    name='PDS_Project',
    version='0.0.1.dev15',
    description="Semester Project - Programming Data Science",
    author="Student",
    author_email="student@uni-koeln.de",
    packages=["nextbike"],
    install_requires=['pandas', 'scikit-learn', 'click', 'geopy'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.cli:main']
    }
)
