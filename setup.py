from setuptools import setup, find_packages
  
setup(
    name='fuse_toolkit',
    version='0.1',
    description='FUSE toolkit supports fluorescent cell image alignment and analysis.',
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