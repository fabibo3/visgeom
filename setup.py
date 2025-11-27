from setuptools import setup, find_packages
setup(
    name='visgeom',
    version='0.1',
    author='Fabian Bongratz',
    packages = find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'visgeom=visgeom.visgeom:main',
            'plot-cortex=visgeom.plot_cortex:main'
        ]
    }
)
