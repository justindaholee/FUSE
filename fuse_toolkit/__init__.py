"""
FUSE toolkit (fuse_toolkit)

The FUSE toolkit provides a set of tools for processing and analyzing
fluorescent cell images, including functions for frame-by-frame analysis,
image processing, and signal derivation.
"""

__version__ = "0.1.0b7"
__author__ = 'Shani Zuniga'
__credits__ = 'Berndt Lab, University of Washington'

from .frame_by_frame import frame_by_frame # noqa: F401
from .img_processing import (
    read_multiframe_tif, # noqa: F401
    process_image, # noqa: F401
    extract_cells, # noqa: F401
    rearrange_dimensions # noqa: F401
)
from .lineage_management import Library # noqa: F401
from .signal_derivation import get_signal  # noqa: F401