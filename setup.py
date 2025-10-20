#!/usr/bin/env python3
"""
Setup script for the vogent-turn package.

Install in development mode:
    pip install -e .

Install normally:
    pip install .
"""

from setuptools import setup, find_packages

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = []
    for line in f:
        line = line.strip()
        # Skip comments, empty lines, and -r references
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name='vogent-turn',
    version='0.1.0',
    description='Lightweight turn detection library for conversational AI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Vogent',
    url='https://github.com/vogent/vogent-turn',
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'vogent-turn-predict=vogent_turn.predict:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    keywords='turn-detection conversation-ai voice speech multimodal whisper',
)
