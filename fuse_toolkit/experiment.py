import os
from pathlib import Path

class Experiment:
    def __init__(self, date, expID, path, parseID, separator, channelInfo, 
                 channelToSegment, numFrames, frameInterval, frameToSegment, Expnote):
        self.date = date
        self.expID = expID
        self.path = path
        self.parseID = parseID
        self.separator = separator
        self.channelInfo = channelInfo
        self.channelToSegment = channelToSegment
        self.numFrames = numFrames
        self.frameInterval = frameInterval
        self.frameToSegment = frameToSegment
        self.Expnote = Expnote

        self.parseID = self.parseID.split(sep=self.separator)
        self.channelInfo = self.channelInfo.split(sep=self.separator)
        self.MultiChannel = len(self.channelInfo) > 1

        if self.frameToSegment != 'all':
            self.frameToSegment = int(self.frameToSegment)

        # Create a new directory and save information about the experiment
        isDirectory = os.path.isdir(self.path)
        if os.path.splitext(self.path)[-1].lower() == '.tif':
            self.folder = os.path.abspath(os.path.join(self.path, os.pardir))
        elif isDirectory:
            self.folder = self.path
        else:
            raise FileNotFoundError("The path does not exist: " + self.path)
            
        #Create a folder where segmentation is saved
        Path(self.folder + "/" + date + "_" + expID).mkdir(parents=True, exist_ok=True)
        Path(self.folder + "/" + date + "_" + expID + "/segmentations").mkdir(
            parents=True, exist_ok=True)

        #Experiment Info saved to text file (if doesn't exist)
        if os.path.exists(self.folder + "/" + date + "_" + expID + '/info_exp.txt'):
            pass
        else:
            f = open(self.folder + "/" + date + "_" + expID + '/info_exp.txt', 'w')
            f.write("date: " + repr(date) + '\n')
            f.write("expID: " + repr(expID) + '\n')
            f.write("path: " + repr(path) + '\n')
            f.write("parseID: " + repr(parseID) + '\n')
            f.write("separator: " + repr(separator) + '\n')
            f.write("MultiChannel: " + repr(self.MultiChannel) + '\n')
            f.write("channelInfo: " + repr(channelInfo) + '\n')
            f.write("numFrames: " + repr(numFrames) + "\n")
            f.write("frameInterval: " + repr(frameInterval) + "\n")
            f.write("Expnote: " + repr(Expnote) + '\n')
            f.close()
