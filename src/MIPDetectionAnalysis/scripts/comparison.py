import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    input_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/pedestals_MIP.csv"
    reconstructed_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/Calibrations/pedestals_MIP.csv"


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



    diff_df['mean'] = ((input_data['pedestal_per_time_sample_mean'] - reconstructed_data['pedestal_per_time_sample_mean']) / input_data['pedestal_per_time_sample_mean']) * 100
    diff_df['std_dev'] = ((input_data['pedestal_per_time_sample_std_dev'] - reconstructed_data['pedestal_per_time_sample_std_dev']) / input_data['pedestal_per_time_sample_std_dev']) * 100

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

    # Stats
    mean_std_dev = round(diff_df['std_dev'].mean(), 6)
    median_std_dev = round(diff_df['std_dev'].median(), 6)
    std_dev_std_dev = round(diff_df['std_dev'].std(), 6)
    len_std_dev = len(diff_df['std_dev'].to_numpy())

    label2 = 'Entires: ' + str(len_std_dev) + '\n' + 'Mean: ' + str(mean_std_dev) + '\n' + 'Median: ' + str(median_std_dev) \
            + '\n' + 'Std Dev: ' + str(std_dev_std_dev)

    diff_df.hist(column=['mean'], bins=100, range=[0, 10], ax=ax, histtype='step', linewidth=2, label=label)
    diff_df.hist(column=['std_dev'], bins=100, range=[-40, 0], ax=ax2, histtype='step', linewidth=2, label=label2)

    ax.set_title(r'Difference of input and reconstructed pedestal $\mu$')
    ax.set_xlabel(r'Difference of input and reconstructed [% of input]')
    ax.set_ylabel("Number of channels")
    ax.legend()

    ax2.set_title(r'Difference of input and reconstructed pedestal $\sigma$')
    ax2.set_xlabel(r'Difference of input and reconstructed [% of input]')
    ax2.set_ylabel("Number of channels")
    ax2.legend()

    plt.show()