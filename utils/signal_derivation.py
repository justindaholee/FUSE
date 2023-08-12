'''
Functions for Deriving Signals

This script provides utility functions for deriving fluorescent signals from
cell images, after they have been processed by FUSE.

Dependencies:

    numpy
    pandas

Functions:

    get_deltaF(df: pd.DataFrame, channel: str, n_frames: int) -> pd.DataFrame:
        Calculates the deltaF/F for each cell in a dataframe.
    get_signal(df: pd.DataFrame, file_names: List[str], naming = List[str],
               separator = str, new_column = 'deltaFoverFo') -> pd.DataFrame:
        Derives specified signal for each cell in a dataframe.

@author: Shani Zuniga
'''
from typing import List

import numpy as np
import pandas as pd

def get_deltaF(df: pd.DataFrame, channel: str = None, n_frames: int = 5
               ) -> pd.DataFrame:
    '''
    Calculates the deltaF/F for each cell in a dataframe.
    
    Args:
        df (pd.DataFrame): A dataframe containing the cell images and metadata.
        channel (str): The channel to use for calculating deltaF/F.
        n_frames (int): The number of frames to use for calculating the baseline.

    Returns:
        pd.DataFrame: A dataframe containing the cell images, metadata, and deltaF/F.
    '''
    if channel is not None:
        df = df[df['Channel'] == channel]
    df = df.dropna()
    
    delta_list = []
    for ID in df['Label'].unique():
        temp_df = df[df['Label'] == ID]
        base_F = temp_df.head(n_frames)['Intensity'].mean()
        temp_df['deltaFoverFo'] = temp_df['Intensity'] / base_F
        delta_list.append(temp_df)
    delta = pd.concat(delta_list, ignore_index=True)
    return delta

def get_signal(df: pd.DataFrame, file_names: List[str], naming = List[str], 
               separator = str, n_frames: int = 5, new_column = 'deltaFoverFo'
               ) -> pd.DataFrame:
    '''
    Calculates the specified signal for each cell in a dataframe.
    Assumes dataframe is output by FUSE engine, generates new column
    for derived signal.
    
    Args
        df (pd.DataFrame): A dataframe containing the cell images and metadata.
        file_names (List[str]): A list of file_names to use for calculating the change.
        naming (List[str]): A list of column names, file naming convention.
        separator (str): The separator used in the file naming convention.
        n_frames (int): The number of frames to use for calculating the baseline.
        new_column (str): The name of the new column.
    
    Returns:
        pd.DataFrame: A dataframe with new column for fluorescent change.
    '''
    # Check for existing deltaFoverFo and initialize deltaF array
    if new_column in df.columns:
        signal_array = df.pop(new_column).values
        signal_array = signal_array.tolist()
    else:
        signal_array = [None] * len(df)

    # Iterate over imgs to plot
    for file_name in file_names:
        # Generate file_df and file_filter to keep relevant info for img
        file_df = df.copy()
        file_filter = [True] * len(df)
        parsed_name = file_name.split(".")[0].split(separator)

        for column, value in zip(naming, parsed_name):
            file_df = file_df[file_df[column].astype(str) == value]
            if column in df.columns:
                col_values = df[column]
                file_filter = file_filter & (col_values == value)

        # Skip iteration if no labels for image
        if file_df['Label'].isna().all():
            continue
    
        for channel in df['Channel'].unique():
            # Make copies of file parsing for each channel
            channel_df = file_df.copy()
            channel_filter = file_filter.copy()
            col_values = df['Channel']
            channel_filter = file_filter & (col_values == channel)

            # Get fluorescent change
            channel_df = get_deltaF(channel_df, channel, n_frames)

            # Merge deltaFoverFo to signal_array
            channel_df = channel_df[[new_column, 'ROI', 'Frame', 'Channel'] 
                                    + naming].copy()
            df = df.merge(channel_df, on=naming + ['ROI', 'Frame', 'Channel'],
                               how='left')
            new_deltaF = df.pop(new_column).values
            new_deltaF = new_deltaF.tolist()
            signal_array = list(np.where(channel_filter, new_deltaF, signal_array))
    
    # Add deltaF array as column of original df
    signal_series = pd.Series(signal_array)
    df[new_column] = signal_series
    return df