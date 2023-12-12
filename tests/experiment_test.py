import os
import glob
import shutil
import pytest
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
        self.expID = 'TestExp'
        self.path = os.path.join('tests', 'test_resources', 'Tester_01.tif')
        self.parseID = 'Name_Well'
        self.separator = '_'
        self.channelInfo = 'RFP_GFP'
        self.channelToSegment = 'RFP'
        self.numFrames = 90
        self.frameInterval = 1
        self.frameToSegment = 0
        self.expNote = 'test note'
        self.test_folder = os.path.join('tests', 'test_resources',
                                        self.date + "_" + self.expID)
        self.json_date = '0000-00-00'
        self.test_json_path = os.path.join('tests', 'test_resources',
                                           self.json_date + "_" + self.expID,
                                           'info_exp.json')
        yield  
        if os.path.isdir(self.test_folder):
            shutil.rmtree(self.test_folder)

        
    def test_initialization(self, setup):
        # Test if the Experiment object is initialized correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.expNote)
        assert experiment.date == self.date
        assert experiment.expID == self.expID
        assert experiment.path == self.path
        assert experiment.parseID == self.parseID.split(sep=self.separator)
        assert experiment.separator == self.separator
        assert experiment.channelInfo == self.channelInfo.split(sep=self.separator)
        assert experiment.channelToSegment == self.channelToSegment
        assert experiment.numFrames == self.numFrames
        assert experiment.frameInterval == self.frameInterval
        assert experiment.frameToSegment == self.frameToSegment
        assert experiment.expNote == self.expNote
        assert hasattr(experiment, 'folder')
        assert hasattr(experiment, 'df')
        if os.path.exists(os.path.join(experiment.folder, f"{experiment.date}_{experiment.expID}", f"{experiment.date}_{experiment.expID}.csv")):
            assert isinstance(experiment.df, pd.DataFrame)
        else:
            assert experiment.df is None


    def test_folder_creation(self, setup):
        # Test if the folder and its subdirectories are created correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.expNote)

        # Main experiment folder
        exp_folder_path = os.path.join(experiment.folder, self.date + '_' + self.expID)
        assert os.path.exists(exp_folder_path) and os.path.isdir(exp_folder_path), "Experiment folder not created correctly."

        # Segmentations subfolder
        segmentations_folder_path = os.path.join(exp_folder_path, 'segmentations')
        assert os.path.exists(segmentations_folder_path) and os.path.isdir(segmentations_folder_path), "Segmentations subfolder not created correctly."

        # Info file
        info_file_path = os.path.join(exp_folder_path, 'info_exp.json')
        assert os.path.exists(info_file_path), "Experiment info file not created correctly."


    def test_info_file_creation(self, setup):
        # Test if the info_exp.json file is created correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.expNote)
        
        info_file_path = os.path.join(experiment.folder, self.date + '_' + self.expID, 'info_exp.json')

        # Check if the file exists
        assert os.path.exists(info_file_path)

        # Check the content of the file
        with open(info_file_path, 'r') as f:
            info_data = json.load(f)

        expected_parseID = self.parseID.split(sep=self.separator)
        expected_channelInfo = self.channelInfo.split(sep=self.separator)
        expected_frameToSegment = int(self.frameToSegment) if self.frameToSegment != 'all' else self.frameToSegment

        assert info_data['date'] == self.date
        assert info_data['expID'] == self.expID
        assert info_data['path'] == self.path
        assert info_data['parseID'] == expected_parseID
        assert info_data['separator'] == self.separator
        assert info_data['multiChannel'] == (len(expected_channelInfo) > 1)
        assert info_data['channelInfo'] == expected_channelInfo
        assert info_data['channelToSegment'] == self.channelToSegment
        assert info_data['numFrames'] == self.numFrames
        assert info_data['frameInterval'] == self.frameInterval
        assert info_data['frameToSegment'] == expected_frameToSegment
        assert info_data['expNote'] == self.expNote


    def test_from_json(self, setup):
        # Test if the Experiment object is initialized correctly from JSON
        experiment = Experiment.from_json(self.test_json_path)

        assert experiment.date == self.json_date
        assert experiment.expID == self.expID
        assert experiment.path == self.path
        assert experiment.parseID == self.parseID.split(self.separator)
        assert experiment.separator == self.separator
        assert experiment.channelInfo == self.channelInfo.split(self.separator)
        assert experiment.channelToSegment == self.channelToSegment
        assert experiment.numFrames == self.numFrames
        assert experiment.frameInterval == self.frameInterval
        assert experiment.frameToSegment == self.frameToSegment
        assert experiment.expNote == self.expNote
        assert experiment.multiChannel == (len(self.channelInfo.split(self.separator)) > 1)

     
    def test_preview_segmentation(self, setup):
        # Test if preview_segmentation() without errors
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.expNote)

        experiment.preview_segmentation(output=False)


    def test_cell_segmentation(self, setup):
        # Tests that cell_segmentation() runs without errors and validates results
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.expNote)

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
            expected_seg_file = os.path.join(experiment.folder, f"{experiment.date}_{experiment.expID}", 
                                             "segmentations", f"{base_name}_seg.tif")
            assert os.path.isfile(expected_seg_file), f"Segmentation file not found for {base_name}"
