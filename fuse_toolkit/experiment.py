import os
import glob
import random
import warnings
from pathlib import Path
import json
import numpy as np
import pandas as pd
from cellpose import models, utils
from skimage import io, measure
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.callbacks import EarlyStopping

from .img_processing import rearrange_dimensions, show_overlay, read_multiframe_tif, extract_cells
from .lineage_management import Library
from .frame_by_frame import frame_by_frame
from . import signal_derivation


class Experiment:
    """
    A class to represent an experiment, encapsulating all relevant details and 
    functionalities such as initialization from a JSON file, directory management, 
    and experiment information handling.

    Properties:
        date (str): Read-only. The date of the experiment.
        exp_ID (str): Read-only. The experiment ID.
        path (str): Read-only. The file path associated with the experiment.
        parse_ID (list): Read-only. A list of IDs derived from parsing a given string with a separator.
        separator (str): Read-only. The separator used for parsing IDs.
        channel_info (list): Read-only. Information about the channels, split using the separator.
        channel_to_seg (str): Read-only. Information on which channel to segment.
        num_frames (int): Read-only. The number of frames in the experiment.
        frame_interval (float): Read-only. The interval between frames.
        frame_to_seg (str/int): Read-only. The specific frame or 'all' frames to segment.
        exp_note (str): Read-only. Notes associated with the experiment.
        multichannel (bool): Read-only. Flag to indicate if there are multiple channels.
        img_folder (str): Read-only. The folder path for the directory containing the images.
        exp_folder (str): Read-only. The folder path for the experiment directory.
        df (pd.DataFrame): Read-only. The experiment cell data-containing dataframe.
    
    Methods:
        from_json(cls, json_file_path):
            Class method to initialize an object from a JSON file.
        preview_segmentation(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0, 
                             min_size=30,diameter=None, output=True):
            Preview segmentation on a randomly chosen image from the experiment directory.
        segment_cells(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0,
                      min_size=30, diameter=None, export_df=True):
            Performs cell segmentation on images and aggregates results in a DataFrame.
        label_cells(self, to_label='ALL', search_radius=100, min_connectivity=100,
                    iou_weight=0.6, visual_weight=0.4, must_overlap=True, export_df=True):
            Performs frame-by-frame cell labeling on cell ROI data for given w/given params
        get_signal(self, signal_info: dict, export_df=True):
            Calculates the specified signal for each cell in a dataframe.
        export_df(path=None):
            Exports the class's DataFrame of cell properties to a CSV file.
    """
    
    def __init__(self, date, exp_ID, path, parse_ID, separator, channel_info, 
                 channel_to_seg, num_frames, frame_interval, frame_to_seg, exp_note):
        """
        Initialize the Experiment object with the provided parameters, creates
        experiment directory if it doesn't yet exist and creates/updates experiment
        information JSON file; returne experiment object.

        Args:
            date (str): The date of the experiment.
            exp_ID (str): The experiment ID, or name.
            path (str): Path to access with experiment data, to an img or folder of imgs.
            parse_ID (str): A string of IDs to be parsed.
            separator (str): The separator used for parsing IDs.
            channel_info (str): A string of channel information to be split.
            channel_to_seg (str): Information on which channel to segment.
            num_frames (int): The number of frames in the experiment.
            frame_interval (float): The interval between frames.
            frame_to_seg (str): The specific frame or 'all' frames to segment.
            exp_note (str): Notes associated with the experiment.
        """
        # Initialize instance variables
        self.__date = date
        self.__expID = exp_ID
        self.__path = path
        self.__parse_ID = parse_ID.split(sep=separator)
        self.__separator = separator
        self.__channel_info = channel_info.split(sep=separator)
        self.__channel_to_seg = channel_to_seg
        self.__num_frames = num_frames
        self.__frame_interval = frame_interval
        self.__frame_to_seg = self._parse_frame_to_segment(frame_to_seg)
        self.__exp_note = exp_note
        self.__multichannel = len(self.__channel_info) > 1

        # Determine and create experiment folder and directories
        self.__img_folder = self._determine_img_folder()
        self.__exp_folder = self._create_experiment_directory()

        # Check for existing segmentation CSV file and load it
        self.__df = self._load_existing_segmentation()

    @classmethod
    def from_json(cls, json_file_path):
        """Given a path to experiment information JSON file, Experiment is initiated and returned"""
        if not Path(json_file_path).is_file():
            raise FileNotFoundError(f"No such file: {json_file_path}")

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        separator = data.get('separator', '')
        parse_ID = separator.join(data.get('parse_ID', []))
        channel_info = separator.join(data.get('channel_info', []))

        return cls(date=data.get('date'),
                   exp_ID=data.get('exp_ID'),
                   path=data.get('path'),
                   parse_ID=parse_ID,
                   separator=separator,
                   channel_info=channel_info,
                   channel_to_seg=data.get('channel_to_seg', ''),
                   num_frames=data.get('num_frames'),
                   frame_interval=data.get('frame_interval'),
                   frame_to_seg=str(data.get('frame_to_seg', 'all')),
                   exp_note=data.get('exp_note'))

    @property
    def date(self):
        return self.__date

    @property
    def exp_ID(self):
        return self.__expID

    @property
    def path(self):
        return self.__path

    @property
    def parse_ID(self):
        return self.__parse_ID

    @property
    def separator(self):
        return self.__separator

    @property
    def channel_info(self):
        return self.__channel_info

    @property
    def channel_to_seg(self):
        return self.__channel_to_seg

    @property
    def num_frames(self):
        return self.__num_frames

    @property
    def frame_interval(self):
        return self.__frame_interval

    @property
    def frame_to_seg(self):
        return self.__frame_to_seg

    @property
    def exp_note(self):
        return self.__exp_note

    @property
    def multichannel(self):
        return self.__multichannel

    @property
    def img_folder(self):
        return self.__img_folder

    @property
    def exp_folder(self):
        return self.__exp_folder

    @property
    def df(self):
        return self.__df


    def preview_segmentation(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0, 
                             min_size=30,diameter=None, output=True):
        """
        Randomly selects an image from the experiment directory, runs segmentation on
        one frame with given parameter settings, and displays resulting masks. Returns
        Cellpose model used for segmentation for further experiment use.

        Args:
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
        files = self._find_image_files(filename_only=False)

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

        del image, masks, flows, styles, diams
        return model


    def segment_cells(self, model_type='cyto2', flow_threshold=0.4, mask_threshold=0.0,
                      min_size=30, diameter=None, export_df=True):
        """
        Performs cell segmentation on all cell images, resulting segmentation masks are
        saved in 'segmentations' folder within the experiment directory, metadata and
        extracted visual features of all ROIs are aggregated into dataframe. The df is
        returned by the funciton and optionally saved in the experiment directory.

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
        files = self._find_image_files(filename_only=False)
        model = self._initialize_model(model_type)
        exp_df = pd.DataFrame()

        for path in tqdm(files, desc='files completed'):
            image = io.imread(path)
            image = rearrange_dimensions(
                image, self.__num_frames, self.__multichannel, self.__channel_info)
            img_to_seg = self._prep_for_seg(image)

            masks, flows, styles, diams = model.eval(img_to_seg, channels=[0, 0],
                                                     flow_threshold=flow_threshold,
                                                     cellprob_threshold=mask_threshold,
                                                     min_size=min_size, diameter=diameter)

            self._export_masks(path, masks)

            file_properties = self._extract_image_properties(path, image, masks)
            del img_to_seg, flows, styles, diams, masks

            file_df = pd.DataFrame(file_properties)
            exp_df = pd.concat([exp_df, file_df], ignore_index=True)

        self.__df = exp_df
        if export_df: 
            self.export_df()
        return exp_df


    def label_cells(self, to_label='ALL', search_radius=100, min_connectivity=100,
                    iou_weight=0.6, visual_weight=0.4, must_overlap=True, export_df=True):
        """
        Performs frame-by-frame cell labeling on cell ROI data for given with the given
        parameters. Populates 'labeling_tools' folder in experiment directory with
        reusable data (encoder model, encoded cells) required for proper labeling. 
        The resulting dataframe with the new 'Label' column is returned by this funciton.
        Optionally saves/updates the resulting dataframe to experiment CSV file.

        Args:
            to_label (str): Specifies images to label ('ALL' or specific filename)
            search_radius (int): Radius, in # pixels, for searching cells
            min_connectivity (int): Minimum connectivity for cells across frames
            iou_weight (float): Weight of intersectoin-over-union in labeling process
            visual_weight (float): Weight of visual characteristics in labeling
            must_overlap (bool): If True, same cells across frame must be overlapping
            
        Returns
            (pd.DataFrame) Updated experiment dataframe with new 'Label' column
        """
        label_list = self._get_column_as_list(column_name='Label')
        imgs_to_label = self._find_image_files(selected=to_label)

        for selected_img in tqdm(imgs_to_label, desc='Processing files'):
            tqdm.write(f"Labeling {selected_img}")
            selected_seg = selected_img.split(".")[0] + "_seg.tif"
            
            masks, imgs_path, masks_path = self._load_masks_and_verify_images(
                selected_seg, selected_img)
            
            file_df, file_filter = self._generate_file_df_and_filter(selected_img)
            
            if self.__frame_to_seg == "all":
                labeling_df = self._generate_labeling_df(file_df)
                cell_vectors = self._load_or_create_cell_vectors(imgs_path, masks_path)
                lib = Library(masks[0], labeling_df)
                lib = frame_by_frame(lib, masks, labeling_df, cell_vectors, search_radius,
                    min_connectivity, iou_weight, visual_weight, must_overlap)
                results = lib.to_dataframe()
                if not results.empty:
                    label_list = self._merge_and_update_labels(
                        label_list, results, file_df, file_filter)
            else:
                self.__df["Label"] = self.__df["ROI"] + 1
                label_list = list(np.where(
                    file_filter, self.__df["Label"].values, label_list))
        self.__df["Label"] = pd.Series(label_list)
        if export_df:
            self.export_df()
        return self.__df


    def get_signal(self, signal_name, signal_params: dict, output_name=None, export_df=True):
        '''
        Calculates the specified signal for each cell in a dataframe.
        Assumes dataframe is output by FUSE engine, generates new column
        for derived signal. The updated df is returned by the function and
        optionally saved in the experiment directory.
        Supported signal types:
            - deltaFoverF0: Measures changes in fluorescence relative to a baseline
            - ratiometric (multichannel only): Intensity ratio between two channels 

        Args:
            signal_name (str): Name of the desired signal to derive (must match a supported
                signal type, case sensitive)
              - "deltaFoverF0"
              - "ratiometric"
            
            signal_params (dict): A dictionary containing the signal parameters,
                contents will depend on the selected signal type:
              - "deltaFoverF0": 
                    - "n_frames" (int): number of frames for baseline (from 0)
                    - "channel" (str): channel to use fluorescent change (match experiment df)
              - "ratiometric": Resulting ratio will be 'channel_1'/'channel_2'
                    - "channel_1" (str): Name of 1st channel (match experiment df)
                    - "channel_2" (str): Name of 2nd channel (math experiment df)
            
            output_name (str): Name of the new column in the df containing the signal
            export_df (bool): If True, exports results to CSV file (default=True)
        Returns:
            pd.DataFrame: A dataframe with new column for fluorescent change.
        '''
        func = self._validate_and_fetch_signal(signal_name, signal_params)
        if not output_name:
            output_name = signal_name
        signal_list = self._get_column_as_list(output_name)

        for filename in self._find_image_files(filename_only=True):
            file_df, file_filter = self._generate_file_df_and_filter(filename)
            if file_df['Label'].isna().all():
                warnings.warn(
                    f"Skipping signal generation for {filename} because cell labeling is not found."
                )
                continue
            new_signal = func(signal=signal_list, df=self.__df, file_filter=file_filter,
                              parse_ID=self.__parse_ID, column_name=output_name, **signal_params)
            signal_list = list(np.where(file_filter, new_signal, signal_list))
        self.__df[output_name] = pd.Series(signal_list)
        if export_df:
            self.export_df()
        return self.__df


    def export_df(self, path=None):
        """
        Exports the class's DataFrame of cell properties to a CSV file.

        Args:
            path (str or None): Destination path for CSV; uses default if None.
        """
        if self.__df is None:
            raise ValueError("Ensure that segmentation has been completed prior to df interaction.")
        if path is None:
            self.__df.to_csv(os.path.join(self.__exp_folder, f"{self.__date}_{self.__expID}.csv"))
        else:
            self.__df.to_csv(path)
        
        
    def _determine_img_folder(self):
        """Identifies folder of interest and returns path"""
        if os.path.isdir(self.__path):
            return self.__path
        elif os.path.splitext(self.__path)[-1].lower() == '.tif':
            return os.path.abspath(os.path.join(self.__path, os.pardir))
        else:
            raise FileNotFoundError(f"The path does not exist: {self.__path}")


    def _create_experiment_directory(self):
        """Creates experiment directory if it doesn't exist"""
        exp_folder = os.path.join(self.__img_folder, f"{self.__date}_{self.__expID}")
        Path(exp_folder).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(exp_folder, "segmentations")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(exp_folder, "labeling_tools")).mkdir(parents=True, exist_ok=True)
        self._save_experiment_info(exp_folder)
        return exp_folder


    def _load_existing_segmentation(self):
        """Checks for an existing segmentation CSV file and loads it if present."""
        csv_path = os.path.join(self.__exp_folder, f"{self.__date}_{self.__expID}.csv")
        if os.path.isfile(csv_path):
            parse_dtypes = {ID: 'str' for ID in self.__parse_ID}
            return pd.read_csv(csv_path, dtype=parse_dtypes)
        return None


    def _save_experiment_info(self, exp_folder):
        """Saves or updates json file with experiment info"""
        info_file = os.path.join(exp_folder, 'info_exp.json')
        current_info = {
            "date": self.__date,
            "exp_ID": self.__expID,
            "path": self.__path,
            "parse_ID": self.__parse_ID,
            "separator": self.__separator,
            "multichannel": self.__multichannel,
            "channel_info": self.__channel_info,
            "channel_to_seg": self.__channel_to_seg,
            "num_frames": self.__num_frames,
            "frame_to_seg": self.__frame_to_seg,
            "frame_interval": self.__frame_interval,
            "exp_note": self.__exp_note
        }
        if os.path.exists(info_file):
            current_info = self._update_experiment_info(info_file, current_info)
        with open(info_file, 'w') as file:
            json.dump(current_info, file, indent=4)
    
    
    def _update_experiment_info(self, info_file, new_info):
        """Updates existing experiment info content; returns updated info"""
        with open(info_file, 'r') as file:
            existing_info = json.load(file)
        for key in new_info:
            if key != 'exp_note':
                if key in existing_info and new_info[key] != existing_info[key]:
                    warnings.warn(
                        f"Updating '{key}' from {existing_info[key]} to {new_info[key]}")
                existing_info[key] = new_info[key]
        if existing_info['exp_note'] != new_info['exp_note']:
            existing_info['exp_note'] += "\n" + new_info['exp_note']
        return existing_info
            

    def _parse_frame_to_segment(self, frame_to_seg):
        """Handles frame_to_seg file format"""
        return int(frame_to_seg) if frame_to_seg != 'all' else frame_to_seg

    
    def _initialize_model(self, model_type):
        """Initializes the segmentation model and returns it"""
        # if Custom_Model:
        #     return models.CellposeModel(gpu='use_GPU', pretrained_model=model_path)
        # elif Omnipose:
        #     return models.CellposeModel(gpu='use_GPU', model_type='cyto2_omni')
        return models.Cellpose(gpu='use_GPU', model_type=model_type)
    
    
    def _find_image_files(self, selected='ALL', filename_only=True):
        """Finds image files in the given path and returns filenames, either as full paths or just the base names."""
        def get_filenames(paths):
            if filename_only:
                return [os.path.basename(path) for path in paths]
            return paths

        if os.path.isdir(self.__path):
            if selected == 'ALL':
                return get_filenames(sorted(glob.glob(os.path.join(self.__path, '*.tif'))))
            else:
                specific_file_path = os.path.join(self.__path, selected)
                if os.path.isfile(specific_file_path) and specific_file_path.endswith('.tif'):
                    return get_filenames([specific_file_path])
                else:
                    raise FileNotFoundError(f"Specified file not found in directory: {specific_file_path}")
        elif os.path.isfile(self.__path) and self.__path.endswith('.tif'):
            if selected == 'ALL' or os.path.basename(self.__path) == selected:
                return get_filenames([self.__path])
            else:
                raise ValueError(f"Specified filename does not match the file in path: {selected}")
        else:
            raise FileNotFoundError(f"No valid .tif files found in path: {self.__path}")
        
        
    def _prepare_sample_image(self, image_path):
        """Isolates and prepares sample image, returns single image"""
        image = io.imread(image_path)
        image = rearrange_dimensions(image, self.__num_frames, self.__multichannel, self.__channel_info)
        image = image[self.__channel_info.index(self.__channel_to_seg)][0]
        return np.squeeze(image)
    
    
    def _prep_for_seg(self, image):
        """Extracts a single channel and/or frame from a multi-dimensional image"""
        if image.ndim == 3:
            image = image
        else:
            image = image[self.__channel_info.index(self.__channel_to_seg)]
        if self.__frame_to_seg != 'all':
            image = [image[self.__frame_to_seg]]
        else:
            image = [i for i in image]
        return image
    
    
    def _export_masks(self, img_path, masks):
        """Exports cell masks to .tif files"""
        name, ftype = os.path.basename(img_path).split(".")
        if len(masks) == 1:
            io.imsave(os.path.join(
                self.__exp_folder, "segmentations", f"{name}_seg.{ftype}"), masks[0])        
        else:
            masks_stack = np.stack(masks)
            io.imsave(os.path.join(
                self.__exp_folder, "segmentations", f"{name}_seg.{ftype}"), masks_stack)
            del masks_stack

        
    def _extract_image_properties(self, img_path, image, masks):
        """Generates list of image and roi properties for the give file/image"""
        parsed_list = img_path[img_path.rfind(os.path.sep)+1:-4].split(sep=self.__separator)
        parsedID = {ID: str(parsed_list[i]) for i, ID in enumerate(self.__parse_ID)}
        
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
                    roi_prop_dict['Time'] = j*self.__frame_interval
                    roi_prop_dict['Channel'] = self.__channel_info[i]
                    roi_prop_dict.update(parsedID)
                    image_props.append(roi_prop_dict)
        return image_props
      

    def _get_column_as_list(self, column_name):
        """Checks for existing df and column; pops existing column and returns it as a list"""
        if self.__df is None:
            raise ValueError("Dataframe 'df' is not initialized. Please run 'segment_cells' first.")
        if column_name in self.__df.columns:
            array = self.__df.pop(column_name).values
            return array.tolist()
        else:
            return [None] * len(self.__df)


    def _load_masks_and_verify_images(self, selected_seg, selected_img):
        """Loads segmentation masks and verifies the existence of corresponding images."""
        masks_path = os.path.join(self.__exp_folder, "segmentations", selected_seg)
        if os.path.exists(masks_path):
            masks = read_multiframe_tif(masks_path)
        else:
            raise FileNotFoundError(f"{masks_path} not found")
        
        imgs_path = os.path.join(self.__img_folder, selected_img)
        if not os.path.exists(imgs_path):
            raise FileNotFoundError(f"{imgs_path} not found")
        
        return masks, imgs_path, masks_path


    def _generate_file_df_and_filter(self, filename):
        """Generates and returns file_df and binary filter to keep relevant img info"""
        img_parsables = os.path.splitext(filename)[0].split(self.__separator)
        filter_dict = dict(zip(self.__parse_ID, img_parsables))
        file_df = self.__df[
            (self.__df['Channel'].astype(str) == self.__channel_to_seg) &
            (self.__df[list(filter_dict)].astype(str) == pd.Series(filter_dict)).all(axis=1)
        ]
        file_filter = (self.__df[list(filter_dict)].astype(str) == pd.Series(filter_dict)).all(axis=1)
        return file_df, file_filter


    def _generate_labeling_df(self, df):
        """Generates and returns df with only labeling-relevant data"""
        labeling_df = df[["Frame", "ROI"]].copy()
        labeling_df["x"] = [coord[0] for coord in df['Centroid']]
        labeling_df["y"] = [coord[1] for coord in df['Centroid']]
        return labeling_df
    
    
    def _load_or_create_cell_vectors(self, imgs_path, masks_path):
        """Loads or generates cell vectors from images and masks; returns the cell vectors"""
        labeling_folder = os.path.join(self.__exp_folder, "labeling_tools")
        vectors_path = os.path.join(
            labeling_folder,
            os.path.splitext(os.path.split(imgs_path)[1])[0] + "_encoded.npz")

        if os.path.exists(vectors_path):
            with np.load(vectors_path) as data_read:
                cell_vectors_dict = {key: data_read[key] for key in data_read.files}
        else:
            channel_idx = self.__channel_info.index(self.__channel_to_seg)
            channel_list = [int(i == channel_idx) for i in range(len(self.__channel_info))]

            cell_dict = extract_cells(imgs_path, masks_path, channel_list)
            encoder = self._load_or_create_encoder(list(cell_dict.values()))
            
            cell_vectors = encoder.predict(
                np.stack(list(cell_dict.values()), axis=0), verbose=0)
            cell_vectors_dict = dict(zip(
                np.array(list(cell_dict.keys())), cell_vectors
                ))
            np.savez(vectors_path, **cell_vectors_dict)
            del cell_dict, encoder, cell_vectors
        return cell_vectors_dict


    def _load_or_create_encoder(self, img_list):
        """Either loads or trains and saves an encoder model from image data."""
        encoder_path = os.path.join(self.__exp_folder, "labeling_tools", "cell_encoder.keras")
        if os.path.exists(encoder_path):
            encoder = keras.models.load_model(encoder_path)
        else:
            encoder = self._generate_and_train_encoder(np.array(img_list))
            encoder.save(encoder_path)
        return encoder


    def _generate_and_train_encoder(self, training_data):
        """Trains an autoencoder on training data and returns the encoder component."""
        x_train = training_data.reshape(training_data.shape[0], 28, 28, 1)
        x_train, x_test = train_test_split(
            x_train, test_size=0.2, random_state=42
        )
        early_stop = EarlyStopping(monitor="val_loss", patience=5)
        encoder = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=[28, 28]),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(30, activation="relu"),
            ]
        )
        decoder = keras.models.Sequential(
            [
                keras.layers.Dense(100, activation="relu", input_shape=[30]),
                keras.layers.Dense(28 * 28, activation="sigmoid"),
                keras.layers.Reshape([28, 28]),
            ]
        )
        autoencoder = keras.models.Sequential([encoder, decoder])
        autoencoder.compile(loss="binary_crossentropy", optimizer="adam")
        autoencoder.fit(
            x_train,
            x_train,
            epochs=100,
            validation_data=[x_test, x_test],
            callbacks=[early_stop],
            verbose=0,
        )
        return encoder


    def _merge_and_update_labels(self, label_list, results_df, file_df, file_filter):
        """Merges result labels to full df and updates it; returns updated label list"""
        results = results_df.rename(columns={"cell_id": "ROI", "lineage_id": "Label"})
        results["ROI"] -= 1
        
        merged_results = file_df.merge(results, on=["ROI", "Frame"], how="left")
        merged_results = merged_results[["Label", "ROI", "Frame"] + self.__parse_ID].copy()
        self.__df = self.__df.merge(
            merged_results, on=self.__parse_ID + ["ROI", "Frame"], how="left")

        new_labels = self.__df.pop("Label").values
        new_labels = new_labels.tolist()
        return list(np.where(file_filter, new_labels, label_list))


    def _validate_and_fetch_signal(self, signal_name, parameters):
        ''' Retrieves the signal function based on type and verifies signal_info'''
        required_params = {
            'deltaFoverF0': ['n_frames', 'channel'],
            'ratiometric': ['channel_1', 'channel_2']}
        signal_processing_functions = {
            'deltaFoverF0': signal_derivation.deltaF,
            'ratiometric': signal_derivation.ratiometric}

        if signal_name not in signal_processing_functions:
            raise ValueError(f"Unsupported signal type: {signal_name}")
        for param in required_params.get(signal_name, []):
            if param not in parameters:
                raise ValueError(f"Missing required parameter for {signal_name}: {param}")
        return signal_processing_functions[signal_name]


