#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

requirements = [
    'scikit-learn',
    'torch',
    'moleculekit',
    'lightning',
    'pdb2pqr'
]

test_requirements = requirements

setup(
    author="Raúl Fernández Díaz",
    author_email='raul.fernandezdiaz@ucdconnect.ie',
    python_requires='>=3.6',
    classifiers=[
    ],
    description="Computational tool for the prediction of metal-binding loading paths in proteins using deep convolutional neural networks.",
    entry_points={
        'console_scripts': [
            'biobrigit=biobrigit.__main__:main',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    # package_data={'': ['*.ckpt', '*.json']},
    data_files=[('',
                 ['biobrigit/utils/stats/gaussian_statistics.json',
                  'biobrigit/utils/stats/residue_statistics.json',
                  'biobrigit/utils/trained_models/BrigitCNN.bak',
                  'README.md'])],
    keywords='biobrigit',
    name='biobrigit',
    packages=find_packages(),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/insillichem/BioBrigit',
    version='0.0.9',
    zip_safe=False,
)
