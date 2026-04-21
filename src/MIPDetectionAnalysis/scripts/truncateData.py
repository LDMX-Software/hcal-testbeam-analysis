import shutil
from collections import OrderedDict
from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import os
import time
import datetime
import random

def select_events(data_df, nbr_events):
    selection = data_df['pf_event'] <= nbr_events
    result = data_df[selection]
    return result

def select_low_event_fraction_channels(data_df, fraction):
    nbr_channels = data_df.groupby(['layer', 'strip']).size()



def select_golden_channels(data_df):
    result = pd.DataFrame()
    for layer in range(1, int(max(data_df['layer'] + 1))):
        if layer % 2 == 0:
            if layer <= 9:
                strip_selection = (data_df['layer'] == layer) & (data_df['strip'] == 6 - 2)
            else:
                strip_selection = (data_df['layer'] == layer) & (data_df['strip'] == 6)
        else:
            if layer <= 9:
                strip_selection = (data_df['layer'] == layer) & (data_df['strip'] == 5 - 2)
            else:
                strip_selection = (data_df['layer'] == layer) & (data_df['strip'] == 5)
        strip_df = data_df[strip_selection]
        result = pd.concat([result, strip_df])
    return result


def select_one_channel(data_df, layer, bar):
    selection = (data_df['layer'] == layer) & (data_df['strip'] == bar)
    return data_df[selection]


def select_middle_channels(data_df):
    result = pd.DataFrame()
    for layer in range(5, int(max(data_df['layer'] -2))):
        if layer % 2 == 0:
            if layer <= 9:
                strip_selection = (data_df['layer'] == layer) & ((data_df['strip'] == 6 - 1) | (data_df['strip'] == 6 - 2) | (data_df['strip'] == 6 - 3))
            else:
                strip_selection = (data_df['layer'] == layer) & ((data_df['strip'] == 6 + 1) | (data_df['strip'] == 6) | (data_df['strip'] == 6 - 1))
        else:
            if layer <= 9:
                strip_selection = (data_df['layer'] == layer) & ((data_df['strip'] == 5 - 1) | (data_df['strip'] == 5 - 2) | (data_df['strip'] == 5 - 3))
            else:
                strip_selection = (data_df['layer'] == layer) & ((data_df['strip'] == 5 + 1) | (data_df['strip'] == 5) | (data_df['strip'] == 5 - 1))
        strip_df = data_df[strip_selection]
        result = pd.concat([result, strip_df])
    return result


if __name__ == '__main__':
    # read in the data file
    data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_strict/strict_two_sided_MIP_efficiencies_adc_mean_with_errors.csv"
    out_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_strict/strict_two_sided_MIP_efficiencies_adc_mean_with_errors_middle_channels.csv"
    data = pd.read_csv(data_file)
    # selected = select_one_channel(data, 1, 2)
    # selected = select_events(data, 1000)
    selected = select_middle_channels(data)
    selected.to_csv(out_file)

