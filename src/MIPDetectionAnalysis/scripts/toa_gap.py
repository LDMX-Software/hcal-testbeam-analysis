import collections
import math
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


def toa_gap(data_df, pedestal_df):
    result_dict = {'layer': [], 'strip': [], 'end': [], 'toa_gap': []}
    data_df.groupby(['layer', 'strip']).apply(lambda group: __toa_gap_group(group, result_dict, pedestal_df))
    return pd.DataFrame(result_dict)


def __toa_gap_group(group, result_dict, pedestal_df):
    layer, strip = group.name
    for end in range(2):
        result_dict['layer'].append(layer)
        result_dict['strip'].append(strip)
        result_dict['end'].append(end)
        pedestal_filter = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (pedestal_df['end'] == end)
        pedestal_filtererd = pedestal_df[pedestal_filter]
        result_dict['toa_gap'].append(min(group['adc_max_end' + str(end)].to_numpy()) - pedestal_filtererd['pedestal_per_time_sample'].iloc[0])


if __name__ == '__main__':
    # read in the data file
    data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal.csv"
    MIP_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/mip_fit_cut_for_range.csv"
    pedestal_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/pedestals_MIP.csv"
    data = pd.read_csv(data_file)
    mip_fits_Data = pd.read_csv(MIP_file)
    pedestal_data = pd.read_csv(pedestal_file)
    selection_toa = (data['toa_end0'] > 0) & (data['toa_end1'] > 0)
    data = data[selection_toa]
    thresholds = toa_gap(data, pedestal_data)
    fig = plt.figure(num=1, clear=True)
    ax = fig.add_subplot()
    mean = thresholds['toa_gap'].mean()
    median = thresholds['toa_gap'].median()
    thresholds.hist('toa_gap', ax=ax, bins=100, label='mean: ' + str(mean) + ', median: ' + str(median))
    ax.set_xlabel('toa gap (ADC)')
    ax.set_ylabel('number of channels')
    ax.set_title('Gap between pedestal and lowest adc max that triggered toa')
    plt.legend()
    plt.show()
    thresholds.to_csv(
        "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/toa.csv")

