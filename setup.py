from setuptools import setup, find_packages
from fuse_toolkit import __version__

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()
  
setup(
    name='fuse_toolkit',
    version=__version__,
    description='FUSE toolkit supports fluorescent cell image alignment and analysis.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Shani Zuniga',
    author_email='shani.zuniga@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'Pillow',
        'tqdm',
        'scipy',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ]
)