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
    author_email='Raul.FernandezDi@autonoma.cat',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
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
    include_package_data=True,
    package_data={'': ['*.ckpt', '*.json', '*.md']},
    keywords='biobrigit',
    name='biobrigit',
    packages=find_packages(include=['biobrigit', 'biobrigit.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/RaulFD-creator/biobrigit',
    version='0.0.1',
    zip_safe=False,
)
