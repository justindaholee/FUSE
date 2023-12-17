"""
FUSE toolkit (fuse_toolkit)

The FUSE toolkit provides a set of tools for processing and analyzing
fluorescent cell images, including functions for frame-by-frame analysis,
image processing, and signal derivation.
"""

__version__ = "0.1.0b14"
__author__ = 'Shani Zuniga'
__credits__ = 'Berndt Lab, University of Washington'

from .experiment import Experiment # noqa: F401
from .img_processing import read_multiframe_tif # noqa: F401