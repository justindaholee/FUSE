import os
import glob
import random
from pathlib import Path
import json
import numpy as np
import pandas as pd
from cellpose import models, utils
from skimage import io, measure
from tqdm.autonotebook import tqdm
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
        df (pd.DataFrame): The experiment cell data-containing dataframe

    Methods:
        from_json(cls, json_file_path):
            Class method to initialize an object from a JSON file.
        preview_segmentation(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0, 
                             min_size=30,diameter=None, output=True):
            Preview segmentation on a randomly chosen image from the experiment directory.
        segment_cells(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0,
                      min_size=30, diameter=None, export_df=True):
            Performs cell segmentation on images and aggregates results in a DataFrame.
        export_df(path=None):
            Exports the class's DataFrame of cell properties to a CSV file.
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

        # Check for existing segmentation CSV file and load it
        self.df = self._load_existing_segmentation()\

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


    def preview_segmentation(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0, 
                             min_size=30,diameter=None, output=True):
        """
        Preview segmentation on a randomly chosen image from the experiment directory.

        Parameters:
            model_type (str): The type of model for Cellpose to use. (default='cyto2')
            flow_threshold (float): Threshold for flow.
            mask_threshold (float): Threshold for mask probability.
            min_size (int): Minimum size of cells to segment.
            diameter (int): Diameter of the cells. If None, a default value is used.
            output (bool): Flag to control output display.

        Returns:
            Cellpose model used in sample segmentation
        """
        model = self._initialize_model(model_type)
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
                f"mask: {mask_threshold}, diameter: {diameter}")
        show_overlay(image, masks, info, os.path.basename(chosen_image_path),
                     utils.outlines_list(masks), show_output=output)

        del image, masks, flows, diams
        return model


    def segment_cells(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0,
                      min_size=30, diameter=None, export_df=True):
        """
        Performs cell segmentation on images and aggregates results in a DataFrame.

        Args:
            model_type (str): The type of model for Cellpose to use. (default='cyto2')
            flow_threshold (float): Threshold for flow.
            mask_threshold (float): Threshold for mask probability.
            min_size (int): Minimum size of cells to segment.
            diameter (int): Diameter of the cells. If None, a default value is used.
            export_df (bool): If True, exports results to CSV

        Returns:
            pd.DataFrame: Resulting df of segmented cell properties.
        """
        model = self._initialize_model(model_type)
        exp_df = pd.DataFrame()

        if os.path.isdir(self.path):
            files = sorted(glob.glob(os.path.join(self.path, '*.tif')))
        elif os.path.splitext(self.path)[-1].lower() == '.tif':
            files = [self.path]

        for path in tqdm(files, desc='files completed'):
            image = io.imread(path)
            image = rearrange_dimensions(
                image, self.numFrames, self.multiChannel, self.channelInfo)
            img_to_seg = self._prep_for_seg(image)

            masks, flows, styles, diams = model.eval(img_to_seg, channels=[0, 0],
                                                     flow_threshold=flow_threshold,
                                                     min_size=min_size, diameter=diameter)

            self._export_masks(path, masks)

            file_properties = self._extract_image_properties(path, image, masks)
            del img_to_seg, flows, styles, diams, masks

            file_df = pd.DataFrame(file_properties)
            exp_df = pd.concat([exp_df, file_df], ignore_index=True)

        self.df = exp_df
        if export_df:
            self.export_df()
            exp_df.to_csv(os.path.join(
                self.folder, f"{self.date}_{self.expID}", f"{self.date}_{self.expID}.csv"))
        
        return exp_df


    def export_df(self, path=None):
        """
        Exports the class's DataFrame of cell properties to a CSV file.

        Args:
            path (str or None): Destination path for CSV; uses default if None.
        """
        if path is None:
            self.df.to_csv(os.path.join(
                self.folder, f"{self.date}_{self.expID}", f"{self.date}_{self.expID}.csv"))
        else:
            self.df.to_csv(path)
        
        
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


    def _load_existing_segmentation(self):
        # Checks for an existing segmentation CSV file and loads it if present.
        csv_path = os.path.join(self.folder, f"{self.date}_{self.expID}", f"{self.date}_{self.expID}.csv")
        if os.path.isfile(csv_path):
            return pd.read_csv(csv_path)
        return None


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

    
    def _initialize_model(self, model_type):
        # Initializes the segmentation model and returns it
        # if Custom_Model:
        #     return models.CellposeModel(gpu='use_GPU', pretrained_model=model_path)
        # elif Omnipose:
        #     return models.CellposeModel(gpu='use_GPU', model_type='cyto2_omni')
        return models.Cellpose(gpu='use_GPU', model_type=model_type)
    
    
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
    
    
    def _prep_for_seg(self, image):
        # Extracts a single channel and/or frame from a multi-dimensional image.
        if image.ndim == 3:
            image = image
        else:
            image = image[self.channelInfo.index(self.channelToSegment)]
        if self.frameToSegment != 'all':
            image = [image[self.frameToSegment]]
        else:
            image = [i for i in image]
        return image
    
    
    def _export_masks(self, img_path, masks):
        # Exports cell masks to .tif files
        name, ftype = os.path.basename(img_path).split(".")
        if len(masks) == 1:
            io.imsave(os.path.join(self.folder, f"{self.date}_{self.expID}", 
                                   "segmentations", f"{name}_seg.{ftype}"), masks[0])        
        else:
            masks_stack = np.stack(masks)
            io.imsave(os.path.join(self.folder, f"{self.date}_{self.expID}", 
                                   "segmentations", f"{name}_seg.{ftype}"), masks_stack)
            del masks_stack

        
    def _extract_image_properties(self, img_path, image, masks):
        # Generates list of image and roi properties for the give file/image
        parsed_list = img_path[img_path.rfind(os.path.sep)+1:-4].split(sep=self.separator)
        parsedID = {ID: str(parsed_list[i]) for i, ID in enumerate(self.parseID)}
        
        file_properties=[]
        for i, channel in enumerate(image):
            channel_prop=[]
            for j, frame in enumerate(channel):
                if len(masks) == 1:
                    channel_prop.append(measure.regionprops(masks[0], frame))
                else:
                    channel_prop.append(measure.regionprops(masks[j], frame))
            file_properties.append(channel_prop)

        image_props = []
        for i, channel_properties in enumerate(file_properties):
            for j, frame_properties in enumerate(channel_properties):
                for k, roi_properties in enumerate(frame_properties):
                    roi_prop_dict = {}
                    roi_prop_dict['Intensity'] = roi_properties.mean_intensity
                    roi_prop_dict['Centroid'] = roi_properties.centroid
                    roi_prop_dict['BB'] = roi_properties.bbox
                    roi_prop_dict['ROI'] = k
                    roi_prop_dict['Frame'] = j
                    roi_prop_dict['Time'] = j*self.frameInterval
                    roi_prop_dict['Channel'] = self.channelInfo[i]
                    roi_prop_dict.update(parsedID)
                    image_props.append(roi_prop_dict)
        return image_props
    
    