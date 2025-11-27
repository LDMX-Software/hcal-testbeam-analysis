import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    input_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_filtered/True_MIP_depth_3_filtered_efficiencies_golden_channels.csv"
    reconstructed_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/MIP_efficiencies/MC_MIP_depth_3_efficiencies_golden_channels.csv"


    with open(input_data_file, 'r') as opened:
        input_data = opened.read()
        input_data = input_data.replace(']', '')
        input_data = input_data.replace('[', '')
    with open(input_data_file, 'w') as opened:
        opened.write(input_data)
    with open(reconstructed_data_file, 'r') as opened:
        reconstructed_data = opened.read()
        reconstructed_data = reconstructed_data.replace(']', '')
        reconstructed_data = reconstructed_data.replace('[', '')
    with open(reconstructed_data_file, 'w') as opened:
        opened.write(reconstructed_data)

    input_data = pd.read_csv(input_data_file)
    reconstructed_data = pd.read_csv(reconstructed_data_file)

    if(len(input_data) <= len(reconstructed_data)):
        reconstructed_data.truncate(before=0, after=len(input_data))
    else:
        input_data.truncate(before=0, after=len(reconstructed_data))

    diff_df = pd.DataFrame()



    diff_df['mean'] = (input_data['MIP_efficiency'] - reconstructed_data['MIP_efficiency'])
    # diff_df['std_dev'] = ((input_data['pedestal_per_time_sample_std_dev'] - reconstructed_data['pedestal_per_time_sample_std_dev']) / input_data['pedestal_per_time_sample_std_dev']) * 100

    print(diff_df)

    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)

    # Stats mean
    mean_mean = round(diff_df['mean'].mean(), 6)
    median_mean = round(diff_df['mean'].median(), 6)
    std_dev_mean = round(diff_df['mean'].std(), 6)
    len_mean = len(diff_df['mean'].to_numpy())

    label = 'Entires: ' + str(len_mean) + '\n' + 'Mean: ' + str(mean_mean) + '\n' + 'Median: ' + str(median_mean) \
            + '\n' + 'Std Dev: ' + str(std_dev_mean)


    diff_df.hist(column=['mean'], bins=100, range=[-0.1, 0.1], ax=ax, histtype='step', linewidth=2, label=label)

    ax.set_title(r'Difference of test beam and MC efficiencies')
    ax.set_xlabel(r'Difference in efficiency')
    ax.set_ylabel("Number of channels")
    ax.legend()


    plt.show()