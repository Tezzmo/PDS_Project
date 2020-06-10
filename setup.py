from setuptools import setup

setup(
    name='PDS_Project',
    version='1.0.1',
    description="Semester Project - Programming Data Science",
    author="Tobias Olbrück, Lorenz Kriehn, Fabian Rehn, Jonas Fröhlich",
    packages=["nextbike"],
    install_requires=['pandas==1.0.3', 'scikit-learn==0.22.2.post1', 'click==7.1.1', 'geopy==1.22.0', 'folium==0.10.1','joblib==0.14.1','matplotlib==3.1.3','seaborn==0.10.1'],
    entry_points={
        'console_scripts': ['nextbike=nextbike.start:main']
    }
)
