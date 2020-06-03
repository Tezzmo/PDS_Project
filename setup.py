from setuptools import setup

setup(
    name='PDS_Project',
    version='0.0.1.dev20',
    description="Semester Project - Programming Data Science",
    author="Student",
    author_email="student@uni-koeln.de",
    packages=["nextbike"],
    install_requires=['pandas', 'scikit-learn', 'click', 'geopy', 'folium','sklearn'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.start:main']
    }
)
