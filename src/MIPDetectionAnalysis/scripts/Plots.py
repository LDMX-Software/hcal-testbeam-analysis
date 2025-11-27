import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import os

results_to_file = False


def makeLabel():
    hep.cms.text(exp="Experiment", text="Internal", fontsize=11, loc=0)


if __name__ == '__main__':
    # plotting setup
    plt.get_backend()

    # Plot style setup
    plt.style.use(hep.style.ROOT)

    # Make figures 3.5 inches wide
    figureWidth = 3.5

    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelweight'] = 'bold'

    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['ytick.major.size'] = 5

    mpl.rcParams['legend.fontsize'] = 8

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Let's choose a consistent color scheme for data and MC
    dataColor = colors[0]
    mcColor = 'black'
    oddColor = colors[1]
    oddColor2 = colors[2]

    print(colors)


    file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/pedestals_no_beam.csv"
    half_ped_file1 = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/no_beam_calibrations/pedestals_no_beam_run_20220424_220632.csv"
    half_ped_file2 = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/no_beam_calibrations/pedestals_no_beam_run_20220424_220542.csv"
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

    half_ped_data1 = pd.read_csv(half_ped_file1)
    half_ped_data2 = pd.read_csv(half_ped_file2)
    concat_ped_data = pd.concat([half_ped_data1, half_ped_data2])

    merged = concat_ped_data.merge(data_MIP_parameters, on=['layer', 'strip', 'end'], how='left')
    concat_ped_data['ped_diff'] = (merged['mpv']/8) / merged['pedestal_per_time_sample_std_dev']
    mean = concat_ped_data['ped_diff'].mean()
    # data['ped_diff'] = (data_MIP_parameters['mpv'] - data['pedestal_per_time_sample_mean']) / data['pedestal_per_time_sample_std_dev']
    temp = data_MIP['low']
    concat_ped_data['pedestal'] = concat_ped_data['pedestal_per_time_sample_mean'] * 8


    # Standard deviation in MIP equivalents
    concat_ped_data['std_dev_mip_eq'] = 8 * merged['pedestal_per_time_sample_std_dev'] / (merged['mpv'])
    mean_mip_eq = round(concat_ped_data['std_dev_mip_eq'].mean(), 6)
    median_mip_eq = round(concat_ped_data['std_dev_mip_eq'].median(), 6)
    std_dev_mip_eq = round(concat_ped_data['std_dev_mip_eq'].std(), 6)
    len_mip_eq = len(concat_ped_data['std_dev_mip_eq'].to_numpy())

    mean_ped = round(concat_ped_data['pedestal_per_time_sample_mean'].mean(), 6)
    median_ped = round(concat_ped_data['pedestal_per_time_sample_mean'].median(), 6)
    std_dev_ped = round(concat_ped_data['pedestal_per_time_sample_mean'].std(), 6)
    len_ped = len(concat_ped_data['pedestal_per_time_sample_mean'].to_numpy())

    label = 'Entires: ' + str(len_mip_eq) + '\n' + 'Mean: ' + str(mean_mip_eq) + '\n' + 'Median: ' + str(median_mip_eq)\
            + '\n' + 'Std Dev: ' + str(std_dev_mip_eq)
    label2 = 'Entires: ' + str(len_ped) + '\n' + 'Mean: ' + str(mean_ped) + '\n' + 'Median: ' + str(median_ped) \
            + '\n' + 'Std Dev: ' + str(std_dev_ped)


    fig2, ax2 = plt.subplots(1, 1)

    fig2.set_size_inches(figureWidth, 1.75)
    bins = np.linspace(0.0, 6, 40 + 1)
    data_col = concat_ped_data['std_dev_mip_eq'].to_numpy().flatten()
    data_col *= 100

    n, bins, patches = plt.hist(data_col, bins, histtype='stepfilled', linewidth=2, label='Data', color=dataColor)
    n = np.append(n, np.nan)
    hist_dict = {'bins': bins, 'data': n}
    hist_df = pd.DataFrame(hist_dict)
    hist_df.to_csv("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/plots/histogram_data/pedestals_noise_hist.csv")

    ax2.set_xlim(min(bins), max(bins))
    ax2.set_ylim(0, 110)
    ax2.set_xlabel("Noise σ [% MIP Equivalents]")
    ax2.set_ylabel("# of Channels")
    ax2.set_title('')
    makeLabel()

    plt.legend(loc=0)
    plt.savefig("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/plots/paper_plots/PedestalPeakDistanceTest.pdf", bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)

    hist = data.hist(column=['pedestal_per_time_sample_mean'], bins=bins, range=[80, 170], ax=ax, label=label2, histtype='step', linewidth=2)
    # data.hist(column=['ped_diff'], bins=90, ax=ax3, label=('mean: ' + str(mean)))
    ax.set_title("Fitted pedestal mean Test beam")
    ax.set_xlabel(r'Fitted $\mu$ [mean ADC]')
    ax.set_ylabel("Number of channels")

    ax3.set_title("Distance between pedestal peak and MIP peak")
    ax3.set_xlabel("Pedestal standard deviations")
    ax3.set_ylabel("Number of channels")
    ax2.legend()
    ax.legend()

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