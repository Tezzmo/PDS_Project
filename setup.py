from setuptools import setup

setup(
    name='PDS_Project',
    version='1.0.1',
    description="Semester Project - Programming Data Science",
    author="Tobias Olbrück, Lorenz Kriehn, Fabian Rehn, Jonas Fröhlich",
    packages=["nextbike"],
    install_requires=['pandas', 'scikit-learn', 'click', 'geopy', 'folium','sklearn'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.start:main']
    }
)
