'''
Functions for Deriving Signals

This script provides specific signal functions for deriving fluorescent signals from
cell images, after they have been processed by FUSE. 
Supported signal types and required parameters:
    - 'deltaFoverF0': Requires 'n_frames', number of frames for baseline.
    - 'ratiometric(multichannel only)': Requires 'channel_1', 'channel_2',
        names of the channels for ratio such that: 'channel_1'/'channel_2'
Accessed by the Experiment.get_signal() function.

Dependencies:

    numpy
    pandas

Functions:

    deltaF(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
           parse_ID: List[str], column_name: str, channel: str, n_frames: int =  5
           ) -> List[float]:
        Calculates the deltaF/F for each cell for a specific file and updates the
        signal list.
    ratiometric(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
                parse_ID: List[str], column_name: str, channel_1: str, channel_2: str
                ) -> List[float]:
        Calculates the ratiometric signals in cells for a specific file and updates 
        the signal list.
    
@author: Shani Zuniga
'''
from typing import List

import numpy as np
import pandas as pd

def deltaF(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
           parse_ID: List[str], column_name: str, channel: str, n_frames: int =  5
           ) -> List[float]:
    '''
    Calculates the deltaF/F for each cell in a dataframe for a specific file and updates
    the signal list.
    
    Args:
        signal (List[float]): A list of the signal values to be updated.
        df (pd.DataFrame): A dataframe containing the cell images and metadata.
        file_filter (List[bool]): A boolean filter for the specified file in the df.
        parse_ID (List[str]): A list of column names, file naming convention.
        column_name (str): The name of the column used for deltaF signal in the df.
        channel (str): The channel to use for calculating deltaF/F.
        n_frames (int): The number of frames to use for calculating the baseline.

    Returns:
        List[float]: A list of the updated signal values.
    '''
    file_df = df[file_filter].dropna(subset=['Label']).copy()
    
    for ch in file_df['Channel'].unique():
        for ID in file_df[file_df['Channel'] == ch]['Label'].unique():
            cell_df = file_df[(file_df['Label'] == ID) & (file_df['Channel'] == ch)]
            base_F = cell_df.head(n_frames)['Intensity'].mean()
            file_df.loc[cell_df.index, column_name] = cell_df['Intensity'] / base_F

    merge_cols = ['ROI', 'Frame', 'Channel'] + parse_ID
    file_df = file_df[merge_cols + [column_name]].drop_duplicates(merge_cols)
    merged_df = df.merge(file_df, on=merge_cols, how='left')

    channel_filter = (df['Channel'] == channel) & file_filter
    updated_signal = np.where(channel_filter, merged_df[column_name].values, signal)
    return updated_signal.tolist()


def ratiometric(signal: List[float], df: pd.DataFrame, file_filter: List[bool],
                parse_ID: List[str], column_name: str, channel_1: str, channel_2: str
                ) -> List[float]:
    '''
    Calculates the ratiometric signal for each cell in a dataframe
    for a specific file and updates the signal list.
    
    Args:
        signal (List[float]): A list of the signal values to be updated.
        df (pd.DataFrame): A dataframe containing the cell images and metadata.
        file_filter (List[bool]): A boolean filter for the specified file in the df.
        parse_ID (List[str]): A list of column names, file naming convention.
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
    
    channel_1_df = df[file_filter & (df['Channel'] == channel_1)].copy()
    channel_1_df = channel_1_df.rename(columns={'Intensity': 'ch1_intensity'})
    channel_1_df = channel_1_df.dropna(subset=['Label'])
    channel_2_df = df[file_filter & (df['Channel'] == channel_2)].copy()
    channel_2_df = channel_2_df.rename(columns={'Intensity': 'ch2_intensity'})
    channel_2_df = channel_2_df.dropna(subset=['Label'])

    merged_df = channel_1_df.merge(
        channel_2_df, on=parse_ID + ['ROI', 'Frame', 'Label'], how='left') 
    merged_df[column_name] = merged_df['ch1_intensity'] / merged_df['ch2_intensity']
    
    df = df.merge(merged_df, on=parse_ID + ['ROI', 'Frame'], how='left')
    new_ratios = df.pop(column_name).values
    updated_signal = list(np.where(file_filter, new_ratios.tolist(), signal))
    return updated_signal