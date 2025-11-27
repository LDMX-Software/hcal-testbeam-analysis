import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

results_to_file = False

if __name__ == '__main__':

    file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/Calibrations/pedestals_MIP.csv"
    dest = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/Simple_MIPs/deviant_channels_MIP"
    file2 = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/mip_fit_cut_for_range.csv"
    file3 = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/mip_fit_parameters.csv"

    with open(file, 'r') as opened:
        data = opened.read()
        data = data.replace(']', '')
        data = data.replace('[', '')
    with open(file, 'w') as opened:
        opened.write(data)

    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)
    data = pd.read_csv(file)
    data_MIP = pd.read_csv(file2)
    data_MIP_parameters = pd.read_csv(file3)
    merged = data.merge(data_MIP_parameters, on=['layer', 'strip', 'end'], how='left')
    data['ped_diff'] = (merged['mpv']/8) / merged['pedestal_per_time_sample_std_dev']
    mean = data['ped_diff'].mean()
    # data['ped_diff'] = (data_MIP_parameters['mpv'] - data['pedestal_per_time_sample_mean']) / data['pedestal_per_time_sample_std_dev']
    temp = data_MIP['low']
    data['pedestal'] = data['pedestal_per_time_sample_mean'] * 8
    data.hist(column=['pedestal'], bins=50, ax=ax)

    # Standard deviation in MIP equivalents
    data['std_dev_mip_eq'] = merged['std_dev'] / merged['mpv']
    mean_mip_eq = round(data['std_dev_mip_eq'].mean(), 6)
    median_mip_eq = round(data['std_dev_mip_eq'].median(), 6)
    std_dev_mip_eq = round(data['std_dev_mip_eq'].std(), 6)
    len_mip_eq = len(data['std_dev_mip_eq'].to_numpy())

    label = 'Entires: ' + str(len_mip_eq) + '\n' + 'Mean: ' + str(mean_mip_eq) + '\n' + 'Median: ' + str(median_mip_eq)\
            + '\n' + 'Std Dev: ' + str(std_dev_mip_eq)
    data.hist(column=['std_dev_mip_eq'], bins=60, range=[0, 0.06], ax=ax2, label=label, histtype='step', linewidth=2)
    # data.hist(column=['ped_diff'], bins=90, ax=ax3, label=('mean: ' + str(mean)))
    ax.set_title("adc pedestal VALUE over all channels")
    ax.set_xlabel("ADC sum")
    ax.set_ylabel("Number of channels")
    ax2.set_title("Fitted pedestal standard deviation Test beam")
    ax2.set_xlabel(r'Fitted $\sigma$ [MIP Equivalent]')
    ax2.set_ylabel("Number of channels")
    ax2.ticklabel_format(useOffset=False, style='plain')
    ax3.set_title("Distance between pedestal peak and MIP peak")
    ax3.set_xlabel("Pedestal standard deviations")
    ax3.set_ylabel("Number of channels")
    ax2.legend()

    high_dev_df = data.loc[data['pedestal_per_time_sample_std_dev'] >= 50]
    print('High deviation channels: \n', high_dev_df[['layer', 'strip', 'end', 'pedestal_per_time_sample_std_dev']])

    high_ped_diff_df = data.loc[abs(data['ped_diff']) >= 5]
    print('High ped diff channels: \n', high_ped_diff_df[['layer', 'strip', 'end', 'ped_diff']])

    # copying over 1D_histograms of deviant channels
    if(results_to_file):
        source = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/Simple_MIPs/pedestals"

        allfiles = os.listdir(source)

        for f in allfiles:
            for i in high_ped_diff_df.index:
                if(f == '/ped_per_time_step_side' + str(high_ped_diff_df.loc[i, 'end']) + '_layer_' + str(high_ped_diff_df.loc[i, 'layer'])
                        + '_bar_' + str(high_ped_diff_df.loc[i, 'strip']) + '.pdf'):
                    src_path = os.path.join(source, f)
                    dest_path = os.path.join(dest, f)
                    shutil.copy(src_path, dest_path)

            for i in high_dev_df.index:
                if(f == '/ped_per_time_step_side' + str(high_dev_df.loc[i, 'end']) + '_layer_' + str(high_dev_df.loc[i, 'layer'])
                        + '_bar_' + str(high_dev_df.loc[i, 'strip']) + '.pdf'):
                    src_path = os.path.join(source, f)
                    dest_path = os.path.join(dest, f)
                    shutil.copy(src_path, dest_path)

    # high_ped_diff_df.to_csv(dest + 'high_ped_diff.csv', index=False)
    # high_dev_df.to_csv(dest + 'high_dev.csv', index=False)

    plt.show()