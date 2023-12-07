import os
import shutil
import pytest
import json
from fuse_toolkit import Experiment

class TestExperimentClass:
    
    @pytest.fixture
    def setup(self):
        # Setup code
        print('Setting up...')
        self.test_folder = os.path.join('tests', 'test_resources', "")
        self.date = '2023-12-31'
        self.expID = 'TestExp'
        self.path = os.path.join('tests', 'test_resources', 'image.tif')
        self.parseID = 'Name_Well'
        self.separator = '_'
        self.channelInfo = 'RFP_GFP'
        self.channelToSegment = 'RFP'
        self.numFrames = 90
        self.frameInterval = 1
        self.frameToSegment = 'all'
        self.expNote = 'test note'
        self.test_folder = os.path.join('tests', 'test_resources',
                                        self.date + "_" + self.expID)
        self.json_date = '0000-00-00'
        self.test_json_path = os.path.join('tests', 'test_resources',
                                           self.json_date + "_" + self.expID,
                                           'info_exp.json')

        yield
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
        assert experiment.frameToSegment == 'all'
        assert experiment.expNote == self.expNote
        assert hasattr(experiment, 'folder')

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
        # Initialize the Experiment object with test data
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.expNote)

        # Test parameters for preview_segmentation
        min_size = 100
        flow_threshold = 0.4
        mask_threshold = 0.0
        rescale = 1
        diameter = 30

        try:
            experiment.preview_segmentation(min_size, flow_threshold, mask_threshold,
                                            rescale, diameter, output=False)
            assert True
        except Exception as e:
            pytest.fail(f"preview_segmentation method failed: {e}")