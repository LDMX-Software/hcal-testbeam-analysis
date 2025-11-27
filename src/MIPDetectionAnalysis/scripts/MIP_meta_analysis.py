import math
import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

import os



def __plot_layer_count_efficiencies(group):
    fig, axes = plt.subplots(2, 1)
    group.plot('strip', 'count', kind='bar', logy=True, ax=axes[0], label='nbr_MIPs')
    group.plot('strip', 'MIP_efficiency', kind='bar', ax=axes[1], label='MIP_efficiency')
    axes[0].set_xlabel('strip')
    axes[1].set_xlabel('strip')
    axes[0].set_ylabel('nbr_MIPs (log)')
    axes[1].set_ylabel('MIP_efficiency')
    fig.suptitle('Event count vs MIP efficiency')
    plt.savefig("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/layerwise_plots/event_count_vs_MIP_efficiency_layer" + str(group.name))
    fig.clear()
    plt.close(fig)


def makeLabel():
    hep.cms.text(exp="Experiment", text="Internal", fontsize=11, loc=0)


if __name__ == '__main__':
    plt.get_backend()
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

    # print(mpl.backends.backend_registry.list_builtin(mpl.backends.BackendFilter.NON_INTERACTIVE))

    full_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal_truncated.csv"
    fileMC = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/MIP_efficiencies/strict_two_sided_MIP_efficiencies_adc_mean_middle_channels.csv"
    fileFilteredData = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_strict/strict_two_sided_MIP_efficiencies_middle_channels.csv"
    fileUnfilteredData = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_strict/strict_two_sided_MIP_efficiencies_adc_mean_with_errors_middle_channels.csv"
    MIP_data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/hcal_true_MIP_filtered.csv"

    with open(fileMC, 'r') as opened:
        dataMC = opened.read()
        dataMC = dataMC.replace(']', '')
        dataMC = dataMC.replace('[', '')
    with open(fileMC, 'w') as opened:
        opened.write(dataMC)
    with open(fileFilteredData, 'r') as opened:
        FilteredData = opened.read()
        FilteredData = FilteredData.replace(']', '')
        FilteredData = FilteredData.replace('[', '')
    with open(fileFilteredData, 'w') as opened:
        opened.write(FilteredData)

    dataMC = pd.read_csv(fileMC)
    FilteredData = pd.read_csv(fileFilteredData)
    UnfilteredData = pd.read_csv(fileUnfilteredData)
    MIP_data = pd.read_csv(MIP_data_file)
    full_data = pd.read_csv(full_data_file)

    dataMC['Two_sided_MIP_efficiency'] = dataMC['Two_sided_MIP_efficiency'].apply(
        lambda x: x if x < 1 else math.floor(x))
    FilteredData['Two_sided_MIP_efficiency'] = FilteredData['Two_sided_MIP_efficiency'].apply(
        lambda x: x if x < 1 else math.floor(x))
    UnfilteredData['Two_sided_MIP_efficiency'] = UnfilteredData['Two_sided_MIP_efficiency'].apply(
        lambda x: x if x < 1 else math.floor(x))

    '''
    if(len(data1) <= len(data2)):
        data2.truncate(before=0, after=len(data1))
    else:
        data1.truncate(before=0, after=len(data2))
    '''

    diff_df = pd.DataFrame()

    # Grouping by 'layer' and 'strip' and counting unique occurrences
    sizes_MIP = MIP_data.groupby(['layer', 'strip']).size()
    sizes_total = full_data.groupby(['layer', 'strip'])['adc_sum_end0'].sum()

    # Merging data
    merger_df = dataMC.set_index(['layer', 'strip'])
    merger_df['MIP_count'] = sizes_MIP
    merger_df['total_count'] = sizes_total

    # Reset index after merging
    merger_df = merger_df.reset_index()

    # Remove any duplicates
    merger_df = merger_df.drop_duplicates(subset=['layer', 'strip'])

    # Adjust the 'strip' (bar) number for layers <= 9
    merger_df['strip'] = merger_df.apply(lambda row: row['strip'] + 2 if row['layer'] <= 9 else row['strip'], axis=1)

    # Split into even and odd layers
    merger_df_even = merger_df[merger_df['layer'] % 2 == 0]
    merger_df_odd = merger_df[merger_df['layer'] % 2 != 0]

    # Create pivot tables
    heatmap_data_MIP_odd = merger_df_odd.pivot(index='layer', columns='strip', values='MIP_count')
    heatmap_data_MIP_even = merger_df_even.pivot(index='layer', columns='strip', values='MIP_count')
    heatmap_data_total_odd = merger_df_odd.pivot(index='layer', columns='strip', values='total_count')
    heatmap_data_total_even = merger_df_even.pivot(index='layer', columns='strip', values='total_count')


    # Normalize the data to [0, 1]
    heatmap_data_MIP_odd /= heatmap_data_MIP_odd.sum().sum()
    heatmap_data_MIP_even /= heatmap_data_MIP_even.sum().sum()
    heatmap_data_total_odd /= heatmap_data_total_odd.sum().sum()
    heatmap_data_total_even /= heatmap_data_total_even.sum().sum()

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))

    # Set normalization to be from 0 to 1 for consistent color mapping
    '''
    norm = Normalize(vmin=0, vmax=1)
    '''

    # Plot heatmaps with normalized color scale
    im1 = ax[0].imshow(heatmap_data_total_odd, cmap="YlGnBu", aspect='auto')
    im2 = ax[1].imshow(heatmap_data_MIP_odd, cmap="YlGnBu", aspect='auto')
    im3 = ax2[0].imshow(heatmap_data_total_even, cmap="YlGnBu", aspect='auto')
    im4 = ax2[1].imshow(heatmap_data_MIP_even, cmap="YlGnBu", aspect='auto')

    ax[0].set_title('total adc distribution')
    ax2[0].set_title('total adc distribution')
    ax[1].set_title('tagged MIPs distribution')
    ax2[1].set_title('tagged MIPs distribution')
    ax[0].set_xlabel('bar nbr')
    ax2[0].set_xlabel('bar nbr')
    ax[1].set_xlabel('bar nbr')
    ax2[1].set_xlabel('bar nbr')
    ax[0].set_ylabel('layer nbr')
    ax2[0].set_ylabel('layer nbr')
    ax[1].set_ylabel('layer nbr')
    ax2[1].set_ylabel('layer nbr')

    fig.suptitle('Odd layers')
    fig2.suptitle('Even layers')


    # Add color bar to each plot (optional but recommended)
    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig2.colorbar(im3, ax=ax2[0])
    fig2.colorbar(im4, ax=ax2[1])

    '''
    grouped = merger_df.groupby(['layer'])
    grouped.apply(__plot_layer_count_efficiencies)
    '''
    filter = merger_df['MIP_count'] > 1
    merger_df = merger_df[filter]
    merger_df.plot('MIP_count', 'Two_sided_MIP_efficiency', kind='scatter', logx=True)
    correlation = merger_df['MIP_count'].corr(merger_df['Two_sided_MIP_efficiency'])
    print('correlation: ', correlation)


    dataColor = colors[0]
    mcColor = 'black'
    unfilColor = 'orange'
    oddColor = colors[1]
    oddColor2 = colors[2]



    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(figureWidth, 2.5)
    # fig2, ax2 = plt.subplots(1, 1)
    bins = np.linspace(98.5, 100, 40 + 1)

    data_col = FilteredData['Two_sided_MIP_efficiency'].to_numpy().flatten()
    MC_col = dataMC['Two_sided_MIP_efficiency'].to_numpy().flatten()
    unfil_col = UnfilteredData['Two_sided_MIP_efficiency'].to_numpy().flatten()
    unfil_col *= 100
    MC_col *= 100
    data_col *= 100

    n, bins, patches = plt.hist(data_col, bins, histtype='stepfilled', linewidth=2, label="Data (Filtered)", color=dataColor)
    n2, bins2, patches2 = plt.hist(MC_col, bins, histtype='step', linewidth=2, label="MC", color=mcColor)
    n3, bins3, patches3 = plt.hist(unfil_col, bins, histtype='step', linewidth=2, label="Data (unfiltered)", color=unfilColor)

    bins_temp = bins[:-1].copy()
    hist_dict = {'bins': bins_temp, 'filtered_data': n, 'MC': n2, 'unfiltered_data': n3}
    hist_df = pd.DataFrame(hist_dict)
    hist_df.to_csv(
        "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/plots/histogram_data/MIP_efficiencies_hist.csv")


    plt.xlim(min(bins), max(bins))
    plt.ylim(0, 35)
    plt.xlabel("MIP Efficiency [%]")
    plt.ylabel("# Of Bars")
    plt.title('')
    makeLabel()


    plt.legend(loc=0)
    plt.savefig("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/plots/paper_plots/MIPEfficiencyTest.pdf", bbox_inches="tight")
    plt.show()

    # ax2.set_title("diff std_dev same channels different data sets")
    # ax2.set_xlabel("ADC")
    # ax2.set_ylabel("Number of channels")

    '''
    selection1 = (data1['MIP_efficiency'] < 0.95)
    deviant_channels1 = data2[selection1]

    deviant_channels1.to_csv("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/Simple_MIPs_filtered/Simple_MIPs_filtered_low_efficiency_channels.csv")

    selection2 = (data2['MIP_efficiency'] < 0.95)
    deviant_channels2 = data2[selection2]

    deviant_channels2.to_csv("C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_filtered/True_MIP_filtered_low_efficiency_channels.csv")

    # correlating MIP efficiency with amount of data
    '''

    plt.show()