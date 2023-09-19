#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from setuptools import find_packages

from setuptools import setup


with open('README.rst', 'rt') as readme_file:
    readme = readme_file.read()


def prerelease_local_scheme(version):
    """
    Return local scheme version unless building on master in CircleCI.

    This function returns the local scheme version number
    (e.g. 0.0.0.dev<N>+g<HASH>) unless building on CircleCI for a
    pre-release in which case it ignores the hash and produces a
    PEP440 compliant pre-release version number (e.g. 0.0.0.dev<N>).
    """
    from setuptools_scm.version import get_local_node_and_date

    if os.getenv('CIRCLE_BRANCH') in {'master'}:
        return ''
    else:
        return get_local_node_and_date(version)


setup(
    name='Ftx_sc',
    use_scm_version={'local_scheme': prerelease_local_scheme},
    description='Morphometric feature extraction codes for sub-compartments in histology images',
    long_description=readme,
    long_description_content_type='text/x-rst',
    author='Sam Border',
    author_email='samuel.border@medicine.ufl.edu',
    url='https://github.com/SarderLab/Feature_Extraction_SubCompartments',
    packages=find_packages(exclude=['tests', '*_test']),
    package_dir={
        'Ftx_sc': 'Ftx_sc',
    },
    include_package_data=True,
    install_requires=[
        # scientific packages
        'nimfa>=1.3.2',
        'numpy==1.19.5',
        'scipy>=0.19.0',
        'Pillow==9.5.0',
        'pandas==1.1.5',
        'imageio>=2.3.0',
        'shapely[vectorized]',
        'matplotlib',
        'pyvips',
        'termcolor',
        'opencv-python',
        'scikit-image==0.19.2',
        'scikit-learn==1.0.2',
        'joblib==1.1.0',
        'tifffile==2021.11.2',
        'tiffslide==1.3.0',
        'tqdm==4.64.0',
        'openpyxl',
        'xlrd<2',
        # dask packages
        'dask[dataframe]>=1.1.0',
        'distributed>=1.21.6',
        # large image sources
        'girder-slicer-cli-web',
        'girder-client',
        # cli
        'ctk-cli',
        'Shapely==1.8.5.post1',
        'wsi-annotations-kit==1.2.15',
    ],
    license='Apache Software License 2.0',
    keywords='Ftx_sc',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    zip_safe=False,
)