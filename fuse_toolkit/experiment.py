import os
import glob
import random
from pathlib import Path
import json
import numpy as np
from cellpose import models, utils
from skimage import io
from fuse_toolkit import rearrange_dimensions, show_overlay

class Experiment:
    """
    A class to represent an experiment, encapsulating all relevant details and 
    functionalities such as initialization from a JSON file, directory management, 
    and experiment information handling.

    Attributes:
        date (str): The date of the experiment.
        expID (str): The experiment ID.
        path (str): The file path associated with the experiment.
        parseID (list): A list of IDs derived from parsing a given string with a separator.
        separator (str): The separator used for parsing IDs.
        channelInfo (list): Information about the channels, split using the separator.
        channelToSegment (str): Information on which channel to segment.
        numFrames (int): The number of frames in the experiment.
        frameInterval (float): The interval between frames.
        frameToSegment (str/int): The specific frame or 'all' frames to segment.
        expNote (str): Notes associated with the experiment.
        multiChannel (bool): Flag to indicate if there are multiple channels.
        folder (str): The folder path for the experiment.

    Methods:
        from_json(cls, json_file_path): Class method to initialize an object from a JSON file.
    """
    
    def __init__(self, date, expID, path, parseID, separator, channelInfo, 
                 channelToSegment, numFrames, frameInterval, frameToSegment, expNote):
        """
        Initialize the Experiment object with the provided parameters.

        Parameters:
            date (str): The date of the experiment.
            expID (str): The experiment ID.
            path (str): The file path associated with the experiment.
            parseID (str): A string of IDs to be parsed.
            separator (str): The separator used for parsing IDs.
            channelInfo (str): A string of channel information to be split.
            channelToSegment (str): Information on which channel to segment.
            numFrames (int): The number of frames in the experiment.
            frameInterval (float): The interval between frames.
            frameToSegment (str): The specific frame or 'all' frames to segment.
            expNote (str): Notes associated with the experiment.
        """
        # Initialize instance variables
        self.date = date
        self.expID = expID
        self.path = path
        self.parseID = parseID.split(sep=separator)
        self.separator = separator
        self.channelInfo = channelInfo.split(sep=separator)
        self.channelToSegment = channelToSegment
        self.numFrames = numFrames
        self.frameInterval = frameInterval
        self.frameToSegment = self._parse_frame_to_segment(frameToSegment)
        self.expNote = expNote
        self.multiChannel = len(self.channelInfo) > 1

        # Determine and create experiment folder and directories
        self.folder = self._determine_folder(path)
        self._create_experiment_directory()


    @classmethod
    def from_json(cls, json_file_path):
        if not Path(json_file_path).is_file():
            raise FileNotFoundError(f"No such file: {json_file_path}")

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        separator = data.get('separator', '')
        parseID = separator.join(data.get('parseID', []))
        channelInfo = separator.join(data.get('channelInfo', []))

        return cls(date=data.get('date'),
                   expID=data.get('expID'),
                   path=data.get('path'),
                   parseID=parseID,
                   separator=separator,
                   channelInfo=channelInfo,
                   channelToSegment=data.get('channelToSegment', ''),
                   numFrames=data.get('numFrames'),
                   frameInterval=data.get('frameInterval'),
                   frameToSegment=str(data.get('frameToSegment', 'all')),
                   expNote=data.get('expNote'))


    def preview_segmentation(self, min_size=30, flow_threshold=0.4,
                             mask_threshold=0.0, rescale=1,
                             diameter=None, Custom_Model=False,
                             model_path=None, Omnipose=False, output=True):
        """
        Preview segmentation on a randomly chosen image from the experiment directory.

        Parameters:
        min_size (int): Minimum size of cells to segment.
        flow_threshold (float): Threshold for flow.
        mask_threshold (float): Threshold for mask probability.
        rescale (float): Rescale factor for the image.
        diameter (int): Diameter of the cells. If None, a default value is used.
        Custom_Model (bool): Flag to use a custom model.
        model_path (str): Path to the custom model, if any.
        Omnipose (bool): Flag to use the Omnipose model.
        output (bool): Flag to control output display.

        Returns:
        None
        """
        model = self._initialize_model(Custom_Model, model_path, Omnipose)
        files = self._find_image_files()

        chosen_image_path = random.choice(files)
        if output:
            print(chosen_image_path)

        image = self._prepare_sample_image(chosen_image_path)
        masks, flows, styles, diams = model.eval(image,
                                                 channels=[0, 0],
                                                 flow_threshold=flow_threshold,
                                                 cellprob_threshold=mask_threshold,
                                                 min_size=min_size,
                                                 diameter=diameter)
        info = (f"min_size: {min_size}, flow: {flow_threshold}, "
                f"mask: {mask_threshold}, rescale: {rescale}, diameter: {diameter}")
        show_overlay(image, masks, info, os.path.basename(chosen_image_path),
                     utils.outlines_list(masks), show_output=output)

        del image, masks, flows, diams


    def _determine_folder(self, path):
        # Identifies folder of interest and returns path
        if os.path.isdir(path):
            return path
        elif os.path.splitext(path)[-1].lower() == '.tif':
            return os.path.abspath(os.path.join(path, os.pardir))
        else:
            raise FileNotFoundError(f"The path does not exist: {path}")


    def _create_experiment_directory(self):
        # Creates experiment directory if it doesn't exist
        exp_folder = os.path.join(self.folder, f"{self.date}_{self.expID}")
        Path(exp_folder).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(exp_folder, "segmentations")).mkdir(parents=True, exist_ok=True)
        self._save_experiment_info(exp_folder)


    def _save_experiment_info(self, exp_folder):
        # Creates json file with experiment info if doesn't exist
        info_file = os.path.join(exp_folder, 'info_exp.json')
        if not os.path.exists(info_file):
            with open(info_file, 'w') as f:
                self._write_experiment_info(f)


    def _write_experiment_info(self, file_object):
        # Writes experiment data to json file
        info_dict = {
            "date": self.date,
            "expID": self.expID,
            "path": self.path,
            "parseID": self.parseID,
            "separator": self.separator,
            "multiChannel": self.multiChannel,
            "channelInfo": self.channelInfo,
            "channelToSegment": self.channelToSegment,
            "numFrames": self.numFrames,
            "frameToSegment": self.frameToSegment,
            "frameInterval": self.frameInterval,
            "expNote": self.expNote
        }
        json.dump(info_dict, file_object, indent=4)


    def _parse_frame_to_segment(self, frameToSegment):
        # Handles frameToSegment file format
        return int(frameToSegment) if frameToSegment != 'all' else frameToSegment

    
    def _initialize_model(self, Custom_Model, model_path, Omnipose):
        # Initializes the segmentation model and returns it
        if Custom_Model:
            return models.CellposeModel(gpu='use_GPU', pretrained_model=model_path)
        elif Omnipose:
            return models.CellposeModel(gpu='use_GPU', model_type='cyto2_omni')
        return models.Cellpose(gpu='use_GPU', model_type='cyto2')
    
    
    def _find_image_files(self):
        # Finds image files in the given path and returns filenames
        if os.path.isdir(self.path):
            return sorted(glob.glob(os.path.join(self.path, '*.tif')))
        elif os.path.splitext(self.path)[-1].lower() == '.tif':
            return [self.path]
        raise FileNotFoundError(f"No image files found in the specified path: {self.path}")

    
    def _prepare_sample_image(self, image_path):
        # Isolates and prepares sample image, returns single image
        image = io.imread(image_path)
        image = rearrange_dimensions(image, self.numFrames, self.multiChannel, self.channelInfo)
        image = image[self.channelInfo.index(self.channelToSegment)][0]
        return np.squeeze(image)