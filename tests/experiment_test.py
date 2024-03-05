import os
import glob
import shutil
import pytest
import warnings
import json
import pandas as pd
from fuse_toolkit import Experiment

class TestExperimentClass:
    @pytest.fixture
    def setup(self):
        # Setup code
        print('Setting up...')
        self.test_folder = os.path.join('tests', 'test_resources', "")
        self.date = '2023-12-31'
        self.exp_ID = 'TestExp'
        self.path = os.path.join('tests', 'test_resources', 'Tester_01.tif')
        self.parse_ID = 'Well_Trial'
        self.separator = '_'
        self.channel_info = 'RFP_GFP'
        self.channel_to_seg = 'RFP'
        self.frame_interval = 1
        self.frame_to_seg = 0
        self.exp_note = 'test note'
        self.test_folder = os.path.join('tests', 'test_resources',
                                        self.date + "_" + self.exp_ID)
        self.json_date = '0000-00-00'
        self.test_json_path = os.path.join('tests', 'test_resources',
                                           self.json_date + "_" + self.exp_ID,
                                           'info_exp.json')
        yield  
        if os.path.isdir(self.test_folder):
            shutil.rmtree(self.test_folder)


    def test_initialization(self, setup):
        # Test if the Experiment object is initialized correctly
        experiment = Experiment(self.date, self.exp_ID, self.path, self.parse_ID, self.separator,
                                self.channel_info, self.channel_to_seg, self.frame_interval, 
                                self.frame_to_seg, self.exp_note)
        assert experiment.date == self.date
        assert experiment.exp_ID == self.exp_ID
        assert experiment.path == self.path
        assert experiment.parse_ID == self.parse_ID.split(sep=self.separator)
        assert experiment.separator == self.separator
        assert experiment.channel_info == self.channel_info.split(sep=self.separator)
        assert experiment.channel_to_seg == self.channel_to_seg
        assert experiment.frame_interval == self.frame_interval
        assert experiment.frame_to_seg == self.frame_to_seg
        assert experiment.exp_note == self.exp_note
        assert hasattr(experiment, 'img_folder')
        assert hasattr(experiment, 'exp_folder')
        assert hasattr(experiment, 'df')
        if os.path.exists(os.path.join(
            experiment.exp_folder, f"{experiment.date}_{experiment.exp_ID}.csv")):
            assert isinstance(experiment.df, pd.DataFrame)
        else:
            assert experiment.df is None


    def test_folder_creation(self, setup):
        # Test if the folder and its subdirectories are created correctly
        experiment = Experiment(self.date, self.exp_ID, self.path, self.parse_ID,
                                self.separator, self.channel_info, self.channel_to_seg,
                                self.frame_interval, self.frame_to_seg, self.exp_note)

        # Main experiment folder
        assert os.path.exists(experiment.exp_folder) and os.path.isdir(
            experiment.exp_folder), "Experiment folder not created correctly."

        # Segmentations subfolder
        segmentations_folder_path = os.path.join(experiment.exp_folder, 'segmentations')
        assert os.path.exists(segmentations_folder_path) and os.path.isdir(
            segmentations_folder_path), "Segmentations subfolder not created correctly."

        # Labeling tool subfolder
        labeling_folder_path = os.path.join(experiment.exp_folder, 'labeling_tools')
        assert os.path.exists(labeling_folder_path) and os.path.isdir(
            labeling_folder_path), "Labeling tool subfolder not created correctly."

        # Info file
        info_file_path = os.path.join(experiment.exp_folder, 'info_exp.json')
        assert os.path.exists(info_file_path), "Experiment info file not created correctly."


    def test_info_file_creation(self, setup):
        # Test if the info_exp.json file is created correctly
        experiment = Experiment(self.date, self.exp_ID, self.path, self.parse_ID,
                                self.separator, self.channel_info, self.channel_to_seg,
                                self.frame_interval, self.frame_to_seg, self.exp_note)
        
        info_file_path = os.path.join(experiment.exp_folder, 'info_exp.json')

        # Check if the file exists
        assert os.path.exists(info_file_path)

        # Check the content of the file
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        expected_parse_ID = self.parse_ID.split(sep=self.separator)
        expected_channel_info = self.channel_info.split(sep=self.separator)
        expected_frame_to_seg = int(self.frame_to_seg) if self.frame_to_seg != 'all' else self.frame_to_seg

        assert info_data['date'] == self.date
        assert info_data['exp_ID'] == self.exp_ID
        assert info_data['path'] == self.path
        assert info_data['parse_ID'] == expected_parse_ID
        assert info_data['separator'] == self.separator
        assert info_data['multichannel'] == (len(expected_channel_info) > 1)
        assert info_data['channel_info'] == expected_channel_info
        assert info_data['channel_to_seg'] == self.channel_to_seg
        assert info_data['frame_interval'] == self.frame_interval
        assert info_data['frame_to_seg'] == expected_frame_to_seg
        assert info_data['exp_note'] == self.exp_note


    def test_modifying_json_after_init(self, setup):
       # Test JSON update with warnings for changed experiment parameters.
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Experiment(self.date, self.exp_ID, self.path, self.parse_ID, self.separator,
                self.channel_info, self.channel_to_seg, self.frame_interval,
                self.frame_to_seg, self.exp_note)
            modified = Experiment(self.date, self.exp_ID, self.path, self.parse_ID, 
                self.separator, self.channel_info, self.channel_to_seg, 
                self.frame_interval + 2, self.frame_to_seg, "This is a secondary note.")
            assert len(w) == 1
            assert "frame_interval" in str(w[-1].message)
        with open(os.path.join(modified.exp_folder, 'info_exp.json'), 'r') as file:
            data = json.load(file)
            assert data['frame_interval'] == self.frame_interval + 2
            assert data['exp_note'].endswith("This is a secondary note.")


    def test_from_json(self, setup):
        # Test if the Experiment object is initialized correctly from JSON
        experiment = Experiment.from_json(self.test_json_path)

        assert experiment.date == self.json_date
        assert experiment.exp_ID == self.exp_ID
        assert experiment.path == self.path
        assert experiment.parse_ID == self.parse_ID.split(self.separator)
        assert experiment.separator == self.separator
        assert experiment.channel_info == self.channel_info.split(self.separator)
        assert experiment.channel_to_seg == self.channel_to_seg
        assert experiment.frame_interval == self.frame_interval
        assert experiment.frame_to_seg == 'all'
        assert experiment.exp_note == self.exp_note
        assert experiment.multichannel == (len(self.channel_info.split(self.separator)) > 1)
        if os.path.exists(os.path.join(experiment.exp_folder,
                                       f"{experiment.date}_{experiment.exp_ID}.csv")):
            assert isinstance(experiment.df, pd.DataFrame)
        else:
            assert experiment.df() is None

     
    def test_preview_segmentation(self, setup):
        # Test if preview_segmentation() without errors
        experiment = Experiment(self.date, self.exp_ID, self.path, self.parse_ID,
                                self.separator, self.channel_info, self.channel_to_seg,
                                self.frame_interval, self.frame_to_seg, self.exp_note)

        experiment.preview_segmentation(output=False)


    def test_cell_segmentation(self, setup):
        # Tests that cell_segmentation() runs without errors and validates results
        experiment = Experiment(self.date, self.exp_ID, self.path, self.parse_ID,
                                self.separator, self.channel_info, self.channel_to_seg,
                                self.frame_interval, self.frame_to_seg, self.exp_note)

        result_df = experiment.segment_cells()

        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty
        assert isinstance(experiment.df, pd.DataFrame)
        assert experiment.df.equals(result_df)
        
        if os.path.isdir(experiment.path):
            input_files = sorted(glob.glob(os.path.join(experiment.path, '*.tif')))
        elif os.path.splitext(experiment.path)[-1].lower() == '.tif':
            input_files = [experiment.path]

        for input_file in input_files:
            base_name = os.path.basename(input_file).split('.')[0]
            expected_seg_file = os.path.join(experiment.img_folder,
                                             f"{experiment.date}_{experiment.exp_ID}", 
                                             "segmentations", f"{base_name}_seg.tif")
            assert os.path.isfile(expected_seg_file), f"Segmentation file not found for {base_name}"


    def test_cell_labeling(self, setup):
        # Tests that the labeling function runs without errors
        experiment = Experiment.from_json(self.test_json_path)
        labeled_df = experiment.label_cells(to_label='ALL', search_radius=100, 
                                            min_connectivity=100, iou_weight=0.6, 
                                            visual_weight=0.4, must_overlap=True, 
                                            export_df=True)
        assert 'Label' in labeled_df.columns
    
        
    def test_signal_derivation(self, setup):
        # Tests that the signal derivation function runs without errors
        experiment = Experiment.from_json(self.test_json_path)
        signal_name = 'deltaFoverF0'
        signal_params = {'n_frames': 5, 'channel': 'GFP'}
        signal_df = experiment.get_signal(signal_name, signal_params, export_df=False)
        assert signal_name in signal_df.columns
        
        signal_name = 'ratiometric'
        signal_params = {'channel_1': 'RFP', 'channel_2': 'GFP'}
        signal_df = experiment.get_signal(signal_name, signal_params, export_df=False)
        assert signal_name in signal_df.columns
        