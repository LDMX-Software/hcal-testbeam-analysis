import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':

    test_beam_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal.csv"
    test_beam_spike_errors = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal_spike_errors.csv"
    test_beam_trigger_errors = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal_late_trigger_errors.csv"
    #MC_bleed_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal_late_trigger_errors.csv"

    test_beam_data = pd.read_csv(test_beam_data_file)
    test_beam_spike_data = pd.read_csv(test_beam_spike_errors)
    test_beam_trigger_data = pd.read_csv(test_beam_trigger_errors)
    # MC_bleed = pd.read_csv(MC_bleed_file)

    # Assume A and B are your dataframes, and you want to group by keys 1, 2, and 3
    group_keys = ['pf_event']  # assuming these are actual column names; adjust if needed

    # Group both DataFrames
    grouped_tb_data = test_beam_data.groupby(group_keys)
    grouped_tb_spike = test_beam_spike_data.groupby(group_keys)
    grouped_tb_trigger = test_beam_trigger_data.groupby(group_keys)

    # Compute lengths of each group
    len_tb_data = grouped_tb_data.size()
    len_tb_spike = grouped_tb_spike.size()
    len_tb_trigger = grouped_tb_trigger.size()
    # len_MC_bleed = grouped_MC_bleed.size()

    # Align both group sizes by index and compute the ratio
    fraction_spike = (len_tb_spike / len_tb_data).reset_index(name='fraction')
    fraction_trigger = (len_tb_trigger / len_tb_data).reset_index(name='fraction')

    fraction_spike.to_csv("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/spike_fraction.csv")
    fraction_trigger.to_csv("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/trigger_fraction.csv")