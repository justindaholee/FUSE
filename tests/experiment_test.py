import os
import shutil
import pytest
from fuse_toolkit import Experiment # Import the Experiment class from your module

class TestExperimentClass:
    
    @pytest.fixture
    def setup(self):
        print("setting up...")
        self.test_folder = os.path.join('tmp', 'test_experiment')
        os.makedirs(self.test_folder)
        self.date = "2023-10-11"
        self.expID = "TestExp"
        self.path = os.path.join(self.test_folder, "test_data")
        os.makedirs(self.path)
        self.parseID = "Name_Well"
        self.separator = "_"
        self.channelInfo = "Fluor"
        self.channelToSegment = "Fluor"
        self.numFrames = 90
        self.frameInterval = 1
        self.frameToSegment = 'all'
        self.Expnote = "test note"
        yield
        print("tearing down...")
        shutil.rmtree('tmp')
    
    def test_initialization(self, setup):
        # Test if the Experiment object is initialized correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.Expnote)
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
        assert experiment.Expnote == self.Expnote
        assert hasattr(experiment, "folder")

    def test_folder_creation(self, setup):
        # Test if the folder is created correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.Expnote)
        assert os.path.exists(experiment.folder)

    def test_info_file_creation(self, setup):
        # Test if the info_exp.txt file is created correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.Expnote)
        info_file_path = os.path.join(experiment.folder, self.date + "_" + self.expID,
                                      "info_exp.txt")
        assert os.path.exists(info_file_path)