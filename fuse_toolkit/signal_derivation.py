'''
Functions for Deriving Signals

This script provides specific signal functions for deriving fluorescent signals from
cell images, after they have been processed by FUSE. 
Supported signal types and required parameters:
    - 'deltaFoverFo': Requires 'n_frames', number of frames for baseline.
    - 'ratiometric(multichannel only)': Requires 'channel_1', 'channel_2',
        names of the channels for ratio such that: 'channel_1'/'channel_2'
Accessed by the Experiment.get_signal() function.

Dependencies:

    numpy
    pandas

Functions:

    _deltaF(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
            naming: List[str], column_name: str = 'deltaFoverFo', channel: str = None,
            n_frames: int =  5) -> List[float]:
        Calculates the deltaF/F for each cell in a dataframe for a specific file and
        updates the signal list.
    _ratiometric(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
                 naming: List[str], column_name: str, channel_1: str, channel_2: str
                 ) -> List[float]:
        Calculates the ratiometric signal for each cell in a dataframe
        for a specific file and updates the signal list.
    
@author: Shani Zuniga
'''
from typing import List

import numpy as np
import pandas as pd

def deltaF(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
           naming: List[str], column_name: str = 'deltaFoverFo', channel: str = None,
           n_frames: int =  5) -> List[float]:
    '''
    Calculates the deltaF/F for each cell in a dataframe for a specific file and updates
    the signal list.
    
    Args:
        signal (List[float]): A list of the signal values to be updated.
        df (pd.DataFrame): A dataframe containing the cell images and metadata.
        file_filter (List[bool]): A boolean filter for the specified file in the df.
        naming (List[str]): A list of column names, file naming convention.
        column_name (str): The name of the column used for deltaF signal in the df.
        channel (str): The channel to use for calculating deltaF/F.
        n_frames (int): The number of frames to use for calculating the baseline.

    Returns:
        List[float]: A list of the updated signal values.
    '''
    for channel in df['Channel'].unique():
        # Make copies of file parsing for each channel
        file_df = df[file_filter].copy()
        channel_filter = file_filter.copy()
        channel_filter = list(file_filter & (df['Channel'] == channel))
        channel_df = file_df[file_df['Channel'] == channel]
        channel_df = channel_df.dropna()
    
        # Get fluorescent change
        delta_list = []
        for ID in channel_df['Label'].unique():
            temp_df = channel_df[channel_df['Label'] == ID]
            base_F = temp_df.head(n_frames)['Intensity'].mean()
            temp_df[column_name] = temp_df['Intensity'] / base_F
            delta_list.append(temp_df)
        file_df = pd.concat(delta_list, ignore_index=True)

        # Merge deltaFoverFo to signal_array
        file_df = file_df[[column_name, 'ROI', 'Frame', 'Channel'] + naming].copy()
        df = df.merge(file_df, on=naming + ['ROI', 'Frame', 'Channel'], how='left')
        new_deltaF = df.pop(column_name).values
        new_deltaF = new_deltaF.tolist()
        updated_signal = list(np.where(channel_filter, new_deltaF, signal))
    return updated_signal

def ratiometric(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
                naming: List[str], column_name: str, channel_1: str, channel_2: str
                ) -> List[float]:
    '''
    Calculates the ratiometric signal for each cell in a dataframe
    for a specific file and updates the signal list.
    
    Args:
        signal (List[float]): A list of the signal values to be updated.
        df (pd.DataFrame): A dataframe containing the cell images and metadata.
        file_filter (List[bool]): A boolean filter for the specified file in the df.
        naming (List[str]): A list of column names, file naming convention.
        channel_1 (str): The channel to use for the numerator.
        channel_2 (str): The channel to use for the denominator.
        column_name (str): The name of the column used for ratiometric signal in the df.
    
    Returns:
        List[float]: A list of the updated signal values.
    '''
    if channel_1 not in df['Channel'].unique():
        raise ValueError(f'Channel {channel_1} not found in file.')
    if channel_2 not in df['Channel'].unique():
        raise ValueError(f'Channel {channel_2} not found in file.')
    
    # Extract the data for the two requested channels from the dataframe
    channel_1_df = df[file_filter & (df['Channel'] == channel_1)].copy()
    channel_1_df = channel_1_df.rename(columns={'Intensity': 'ch1_intensity'})
    channel_1_df = channel_1_df.dropna()
    channel_2_df = df[file_filter & (df['Channel'] == channel_2)].copy()
    channel_2_df = channel_2_df.rename(columns={'Intensity': 'ch2_intensity'})
    channel_2_df = channel_2_df.dropna()

    # Merge the two dataframes
    merged_df = channel_1_df.merge(
        channel_2_df, on=naming + ['ROI', 'Frame', 'Label'], how='left') 
    
    # Calculate the ratio of the two channels
    merged_df[column_name] = merged_df['ch1_intensity'] / merged_df['ch2_intensity']
    
    # Update the dataframe with the new ratio values
    df = df.merge(merged_df, on=naming + ['ROI', 'Frame'], how='left')
    new_ratios = df.pop(column_name).values
    new_ratios = new_ratios.tolist()
    updated_signal = list(np.where(file_filter, new_ratios, signal))
    return updated_signal