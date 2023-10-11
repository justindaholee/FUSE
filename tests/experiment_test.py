import unittest
import os
from fuse_toolkit import Experiment # Import the Experiment class from your module

class TestExperimentClass(unittest.TestCase):
    def setUp(self):
        # Create a temporary folder for testing and set up initial parameters
        self.test_folder = "/tmp/test_experiment"
        os.makedirs(self.test_folder)
        self.date = "2023-09-28"
        self.expID = "TestExp"
        self.path = os.path.join(self.test_folder, "test_data")
        self.parseID = "Name_Well"
        self.separator = "_"
        self.channelInfo = "Fluor"
        self.channelToSegment = "Fluor"
        self.numFrames = 90
        self.frameInterval = 1
        self.frameToSegment = 'all'
        self.Expnote = ""

    def test_initialization(self):
        # Test if the Experiment object is initialized correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.Expnote)
        self.assertEqual(experiment.date, self.date)
        self.assertEqual(experiment.expID, self.expID)
        self.assertEqual(experiment.path, self.path)
        self.assertEqual(experiment.parseID, self.parseID)
        self.assertEqual(experiment.separator, self.separator)
        self.assertEqual(experiment.channelInfo, [self.channelInfo])  # It should be a list
        self.assertEqual(experiment.channelToSegment, self.channelToSegment)
        self.assertEqual(experiment.numFrames, self.numFrames)
        self.assertEqual(experiment.frameInterval, self.frameInterval)
        self.assertEqual(experiment.frameToSegment, 'all')  # 'all' should be converted to string
        self.assertEqual(experiment.Expnote, self.Expnote)
        self.assertTrue(hasattr(experiment, "folder"))

    def test_folder_creation(self):
        # Test if the folder is created correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.Expnote)
        self.assertTrue(os.path.exists(experiment.folder))

    def test_info_file_creation(self):
        # Test if the info_exp.txt file is created correctly
        experiment = Experiment(self.date, self.expID, self.path, self.parseID,
                                self.separator, self.channelInfo, self.channelToSegment,
                                self.numFrames, self.frameInterval, self.frameToSegment,
                                self.Expnote)
        info_file_path = os.path.join(experiment.folder, self.date + "_" + self.expID,
                                      "info_exp.txt")
        self.assertTrue(os.path.exists(info_file_path))

    def tearDown(self):
        # Clean up after testing
        os.rmdir(self.test_folder)  # Remove the temporary test folder

if __name__ == '__main__':
    unittest.main()
