import collections
import math
import shutil
from cProfile import label
from collections import OrderedDict

from fontTools.misc.bezierTools import Intersection

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import scipy.signal as sig
import scipy.stats as stats
from scipy.optimize import curve_fit
import os
import time
import datetime
import random
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


'''This file contains a set of functions that constitutes Pedestal analysis, 
readout error analysis, readout error correlations, and MIP efficiency analysis.
It consists of a large amount of mostly independent functions that operate by
manipulating Pandas DataFrames. For examples of how to use the code, see the 
example scripts.'''



def sum_ends(group):
    '''
    sums the adc of the two ends of a scintillator bar
    :param groupby: groupby object generated from grouping the data by layer and strip
    :returns: the group after manipulation
    '''
    group['sum_ends'] = group['adc_sum_end0'] + group['adc_sum_end1']
    return group


def plot_strip_hists(data_df):
    '''
    plots ADC mean histogram of all readout channels in data_df
    :param data_df: dataframe of data to be plotted
    '''
    for layer in range(1, int(data_df['layer'].max()) + 1):

        layer_selection = data_df['layer'] == layer
        layer_data = data_df[layer_selection]

        for strip in range(int(layer_data['strip'].max() + 1)):

            data_selection = (layer_data['strip'] == strip)
            channel_data = layer_data[data_selection]

            for end in range(2):
                print(channel_data)
                fig = plt.figure(num=1, clear=True)
                ax1 = fig.add_subplot()
                channel_data.hist('adc_mean_end' + str(end), ax=ax1, bins=6000, range=[0, 6000], log=True)
                ax1.set_title("layer: " + str(layer) + " bar: " + str(strip) + " end: " + str(end))
                ax1.set_xlabel("ADC mean")
                ax1.set_ylabel("nbr channels")
                plt.show()


def plot_strip_hists_max_adc(data_df):
    '''
    plots ADC max histogram of all readout channels in data_df
    :param data_df: Pandas DataFrame of data to be plotted
    '''
    for layer in range(1, int(data_df['layer'].max()) + 1):

        layer_selection = data_df['layer'] == layer
        layer_data = data_df[layer_selection]

        for strip in range(int(layer_data['strip'].max() + 1)):

            data_selection = (layer_data['strip'] == strip)
            channel_data = layer_data[data_selection]

            for end in range(2):
                print(channel_data)
                fig = plt.figure(num=1, clear=True)
                ax1 = fig.add_subplot()
                channel_data.hist('adc_max_end' + str(end), ax=ax1, bins=6000, range=[0, 6000], log=True)
                ax1.set_title("layer: " + str(layer) + " bar: " + str(strip) + " end: " + str(end))
                ax1.set_xlabel("ADC max")
                ax1.set_ylabel("nbr channels")
                plt.show()


def plot_strip_hists_pedetsal_fit(data_df, pedestal_df, channel):
    '''
    plots the distribution of fitted pedestal values over the full HCal
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class
    :param pedestal_df: Pandas DataFrame containing data produced by the calculatePedestals.py class
    :param channel: selected channels [layer, strip]
    '''
    layer_selection = data_df['layer'] == channel[0]
    layer_data = data_df[layer_selection]
    data_selection = (layer_data['strip'] == channel[1])
    channel_data = layer_data[data_selection]

    layer_ped_selection = pedestal_df['layer'] == channel[0]
    pedestal_layer_data = pedestal_df[layer_ped_selection]
    bar_ped_selection = pedestal_df['strip'] == channel[1]
    pedestal_channel_data = pedestal_layer_data[bar_ped_selection]

    for end in range(2):
        pedestal_end_selection = pedestal_channel_data['end'] == end
        pedestal_data = pedestal_channel_data[pedestal_end_selection]
        fig = plt.figure(num=1, clear=True)
        ax1 = fig.add_subplot()
        bins = 300
        plot_range = [700, 1000]
        mean = pedestal_data['pedestal_per_time_sample_mean'].iloc[0] * 8
        std_dev = pedestal_data['std_dev'].iloc[0]
        x = np.linspace(plot_range[0], plot_range[1], bins)
        fit = stats.norm.pdf(x, mean, std_dev)
        channel_data.hist('adc_sum_end' + str(end), ax=ax1, bins=bins, range=plot_range, log=True, density=True, histtype='step')
        ax1.plot(x, fit, 'k', linewidth=1, linestyle='--')
        ax1.set_title("layer: " + str(channel[0]) + " bar: " + str(channel[1]) + " end: " + str(end))
        ax1.set_xlabel("ADC sum")
        ax1.set_ylabel("probability density")
        ax1.set_ylim(0.000001, 1)
        plt.show()


def show_selected_region(data_df, region_df):
    '''
    Shows the adc sum histogram of data in the region defined
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class
    :param region_df: Pandas DataFrame defining the region
    :return:
    '''
    for layer in range(1, data_df['layer'].max() + 1):

        layer_selection = data_df['layer'] == layer
        layer_data = data_df[layer_selection]
        layer_selection_region = region_df['layer'] == layer
        layer_region = region_df[layer_selection_region]

        for strip in range(int(layer_data['strip'].max() + 1)):

            data_selection = (layer_data['strip'] == strip)
            channel_data = layer_data[data_selection]
            strip_selection_region = layer_region['strip'] == strip
            channel_region = layer_region[strip_selection_region]

            for end in range(2):
                end_selection_region = channel_region['end'] == end
                end_region = channel_region[end_selection_region]
                print(channel_data)
                fig = plt.figure(num=1, clear=True)
                ax1 = fig.add_subplot()
                channel_data.hist('adc_sum_end' + str(end), ax=ax1, bins=1000, range=[0, 3000], log=True)
                ax1.set_title("layer: " + str(layer) + " bar: " + str(strip) + " end: " + str(end))
                ax1.set_xlabel("ADC sum")
                ax1.set_ylabel("nbr channels")
                ax1.vlines([end_region['lower'].values[0], end_region['upper'].values[0]], 0,
                           max(channel_data['adc_sum_end' + str(end)].values), 'g')
                plt.show()


def plot_group(group):
    layer, strip = group.name
    for end in range(2):
        fig = plt.figure(num=1, clear=True)
        ax1 = fig.add_subplot()
        group.hist('adc_sum_end' + str(end), ax=ax1, bins=1000, range=[0, 2048 * 4], log=True)
        ax1.set_title("layer: " + str(layer) + " bar: " + str(strip) + " end: " + str(end))
        ax1.set_xlabel("ADC sum")
        ax1.set_ylabel("nbr channels")
        plt.show()


def select_region(group):
    layer, strip = group.name
    pedestal_selection = (pedestal_data['layer'] == layer) & (pedestal_data['strip'] == strip)
    channel_pedestal = pedestal_data[pedestal_selection]
    lower_bound = channel_pedestal['pedestal_per_time_sample_mean'] * 8 * 2 * 2 + channel_pedestal["std_dev"] * 2 * 2
    signal_selection = (mip_fits_Data['layer'] == layer) & (mip_fits_Data['strip'] == strip)
    channel_signal = mip_fits_Data[signal_selection]
    upper_bound = channel_signal['lower']
    group_dict = {
        'layer': layer,
        'strip': strip,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    pd.concat(channelwise_data, pd.DataFrame(group_dict))


def ADC_middle_region_definition(pedestal_df, mip_df):
    result = []
    for layer in range(1, pedestal_df['layer'].max() + 1):
        layer_selection = pedestal_df['layer'] == layer
        layer_pedestal = pedestal_df[layer_selection]
        # print('strip max: ', layer_pedestal['strip'].max())
        for strip in range(layer_pedestal['strip'].max() + 1):
            for end in range(2):
                pedestal_selection = (layer_pedestal['strip'] == strip) & (layer_pedestal['end'] == end)
                channel_pedestal = layer_pedestal[pedestal_selection]
                lower_bound = channel_pedestal['pedestal'].values[0] + (channel_pedestal['std_dev'].values[0] * 3)
                mip_selection = (mip_df['layer'] == layer) & (mip_df['strip'] == strip) & (mip_df['end'] == end)
                channel_mip = mip_df[mip_selection]
                upper_bound = channel_pedestal['pedestal'].values[0] + channel_mip['low'].values[0]
                result.append({
                    'layer': layer,
                    'strip': strip,
                    'end': end,
                    'lower': lower_bound,
                    'upper': upper_bound
                })
                # print('layer: ' + str(layer) + ' strip: ' + str(strip) + ' end: ' + str(end))
                # print('Upper: ' + str(upper_bound))
                # print('Lower: ' + str(lower_bound))
    return pd.DataFrame(result)


def ADC_pedestal_region_definition(pedestal_df):
    result = []
    for layer in range(1, pedestal_df['layer'].max() + 1):
        layer_selection = pedestal_df['layer'] == layer
        layer_pedestal = pedestal_df[layer_selection]
        # print('strip max: ', layer_pedestal['strip'].max())
        for strip in range(layer_pedestal['strip'].max() + 1):
            for end in range(2):
                pedestal_selection = (layer_pedestal['strip'] == strip) & (layer_pedestal['end'] == end)
                channel_pedestal = layer_pedestal[pedestal_selection]
                # print('channel pedestal: ', channel_pedestal)
                # lower_bound = channel_pedestal['pedestal'].values[0] - (channel_pedestal['std_dev'].values[0] * 6)
                lower_bound = 0
                upper_bound = channel_pedestal['pedestal'].values[0] + (channel_pedestal['std_dev'].values[0] * 6)
                result.append({
                    'layer': layer,
                    'strip': strip,
                    'end': end,
                    'lower': lower_bound,
                    'upper': upper_bound
                })
                # print('layer: ' + str(layer) + ' strip: ' + str(strip) + ' end: ' + str(end))
                # print('Upper: ' + str(upper_bound))
                # print('Lower: ' + str(lower_bound))
    return pd.DataFrame(result)


def ADC_MIP_region_definition(pedestal_df, mip_df):
    result = []
    for layer in range(1, mip_df['layer'].max() + 1):
        layer_selection_mip = mip_df['layer'] == layer
        layer_mip = mip_df[layer_selection_mip]
        layer_selection_pedestal = pedestal_df['layer'] == layer
        layer_pedestal = pedestal_df[layer_selection_pedestal]
        # print('strip max: ', layer_mip['strip'].max())
        for strip in range(layer_mip['strip'].max() + 1):
            for end in range(2):
                pedestal_selection = (layer_pedestal['strip'] == strip) & (layer_pedestal['end'] == end)
                channel_pedestal = layer_pedestal[pedestal_selection]
                mip_selection = (layer_mip['strip'] == strip) & (layer_mip['end'] == end)
                channel_mip = layer_mip[mip_selection]
                # print('channel MIP: ', channel_mip)
                lower_bound = channel_mip['low'].values[0] + channel_pedestal['pedestal'].values[0]
                upper_bound = channel_mip['upper'].values[0] + channel_pedestal['pedestal'].values[0]
                result.append({
                    'layer': layer,
                    'strip': strip,
                    'end': end,
                    'lower': lower_bound,
                    'upper': upper_bound
                })
                # print('layer: ' + str(layer) + ' strip: ' + str(strip) + ' end: ' + str(end))
                # print('Upper: ' + str(upper_bound))
                # print('Lower: ' + str(lower_bound))
    return pd.DataFrame(result)


def ADC_region_definition(pedestal_df, mip_df, lower, upper):
    result = []
    for layer in range(1, pedestal_df['layer'].max() + 1):
        layer_selection = pedestal_df['layer'] == layer
        layer_pedestal = pedestal_df[layer_selection]
        # print('strip max: ', layer_pedestal['strip'].max())
        for strip in range(layer_pedestal['strip'].max() + 1):
            for end in range(2):
                pedestal_selection = (layer_pedestal['strip'] == strip) & (layer_pedestal['end'] == end)
                channel_pedestal = layer_pedestal[pedestal_selection]
                lower_bound = lower
                mip_selection = (mip_df['layer'] == layer) & (mip_df['strip'] == strip) & (mip_df['end'] == end)
                channel_mip = mip_df[mip_selection]
                upper_bound = upper
                result.append({
                    'layer': layer,
                    'strip': strip,
                    'end': end,
                    'lower': lower_bound,
                    'upper': upper_bound
                })
                # print('layer: ' + str(layer) + ' strip: ' + str(strip) + ' end: ' + str(end))
                # print('Upper: ' + str(upper_bound))
                # print('Lower: ' + str(lower_bound))
    return pd.DataFrame(result)


def filter_on_adc_region(data_df, region_df):
    result_df = pd.DataFrame()
    for layer in range(1, region_df['layer'].max() + 1):

        layer_selection_region = (region_df['layer'] == layer)
        layer_region = region_df[layer_selection_region]
        layer_selection_data = (data_df['layer'] == layer)
        layer_data = data_df[layer_selection_data]

        for strip in range(layer_region['strip'].max() + 1):
            strip_selection_data = (layer_data['strip'] == strip)
            channel_data = layer_data[strip_selection_data]
            strip_selection_region = (layer_region['strip'] == strip)
            strip_region = layer_region[strip_selection_region]

            # print('layer: ' + str(layer) + ' strip: ' + str(strip))

            channel_selection_region_0 = (strip_region['end'] == 0)
            channel_selection_region_1 = (strip_region['end'] == 1)
            channel_region_0 = strip_region[channel_selection_region_0]
            channel_region_1 = strip_region[channel_selection_region_1]
            # print('region: ', channel_region_1)
            region_selection = ((channel_region_0['lower'].values[0] <= channel_data['adc_sum_end0']) & (
                    channel_data['adc_sum_end0'] <= channel_region_0['upper'].values[0])) & (
                                       (channel_region_1['lower'].values[0] <= channel_data['adc_sum_end1']) & (
                                       channel_data['adc_sum_end1'] <= channel_region_1['upper'].values[0]))
            selected_data = channel_data[region_selection]

            result_df = pd.concat([result_df, selected_data])
            # print(selected_data)
            # selected_data.hist('adc_sum_end1')
            plt.show()

    return result_df


def plot_pulse_shape_random_event(group, nbr_events=4):
    '''
    plots the pulse shapes of random events
    :param group: groupby object generated from grouping the data by layer and strip
    :param nbr_events: number of events to be ploted
    '''
    layer, strip = group.name
    print('group: layer' + str(layer) + ' strip' + str(strip))
    # Select events to plot
    n = len(group['pf_event'].values)
    if nbr_events > n:
        print('Requested plotting of more events than available, plotting all available events')
        nbr_events = n
    nums = list(range(0, n))
    random.shuffle(nums)
    print('Number of available events: ', n)
    print('Randomly selected event indices: ', nums[:nbr_events])

    plot_events_df = pd.DataFrame()
    for i in range(nbr_events):
        plot_events_df = pd.concat([plot_events_df, group.iloc[[nums[i]]]])
        print('Event dataframe after concatenation: \n', plot_events_df)

    fig, axes = plt.subplots(nbr_events, 2, figsize=(10, 5 * nbr_events))
    for i in range(nbr_events):
        if i >= len(plot_events_df):
            print(f"Skipping event {i} because it's out of bounds")
            continue

        event = plot_events_df.iloc[[i]]
        if event.empty:
            print(f"Event {i} is empty, skipping")
            continue

        end0 = []
        end1 = []
        x = []
        for j in range(8):
            adc_end0_col = f'adc_{j}_end0'
            adc_end1_col = f'adc_{j}_end1'

            if adc_end0_col not in event or adc_end1_col not in event:
                print(f"Columns '{adc_end0_col}' or '{adc_end1_col}' do not exist in event {i}, skipping")
                continue

            end0_val = event[adc_end0_col].values[0]
            end1_val = event[adc_end1_col].values[0]

            if np.isnan(end0_val) or np.isnan(end1_val):
                print(f"NaN value found in '{adc_end0_col}' or '{adc_end1_col}' for event {i}, skipping")
                continue

            end0.append(end0_val)
            end1.append(end1_val)
            x.append(j + 1)

        if not end0 or not end1:
            print(f"No valid data for event {i}, skipping plot")
            continue

        print('end0: ', end0)
        print('end1: ', end1)

        try:
            axes[i][0].plot(x, end0, '*', linestyle='--', linewidth=1)
            axes[i][1].plot(x, end1, '*', linestyle='--', linewidth=1)
            # axes[i][0].set_ylim([0, 200])
            # axes[i][1].set_ylim([0, 200])
            axes[i][0].set_xlabel('time (ordered datapoint)')
            axes[i][1].set_xlabel('time (ordered datapoint)')
            axes[i][0].set_ylabel('adc')
            axes[i][1].set_ylabel('adc')
            # axes[i][0].set_title('event ' + str(event['pf_event'].values[0]) + ' end 0')
            # axes[i][1].set_title('event ' + str(event['pf_event'].values[0]) + ' end 1')
        except Exception as e:
            print(f"Error plotting event {i}: {e}")

    fig.suptitle('Layer ' + str(layer) + ' bar ' + str(strip))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    fig.suptitle('Abberant pulse shapes layer ' + str(layer) + ' bar ' + str(strip))
    plt.show()


def select_golden_channels(data_df):
    '''
    Selects the "Golden channels" of the HCal prototype tested at CERN in 2022
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class
    :return: The manipulated DataFrame
    '''
    result = pd.DataFrame()
    for layer in range(1, max(data_df['layer'] + 1)):
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


def select_faulty_data(data_df, pedestal_df, spike_filter=True, late_trigger_filter=True):
    '''
    NOTE! OUTDATED!
    Method for filtering out readout errors in the data
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class
    :param pedestal_df: Pandas DataFrame containing data produced by the calculatePedestals.py class
    :param spike_filter: Boolean that toggles if the function filters out spike errors
    :param late_trigger_filter: Boolean that toggles if the function filters out spike errors
    :return: faulty_data_indecies, nbr_faulty_pulses A list of the indicies of where readout errors are found in the data and
    the number of errors found
    '''
    # loop through all events in the dataframe
    print('filtering faulty events...')
    faulty_data_indecies = []
    for index, row in data_df.iterrows():

        # extract the pulse shape for the event
        end0 = []
        end1 = []

        for j in range(8):

            adc_end0_col = f'adc_{j}_end0'
            adc_end1_col = f'adc_{j}_end1'

            if adc_end0_col not in row or adc_end1_col not in row:
                print(f"Columns '{adc_end0_col}' or '{adc_end1_col}' do not exist in event {row['pf_event']}, skipping")
                continue

            end0_val = row[adc_end0_col]
            end1_val = row[adc_end1_col]

            if np.isnan(end0_val) or np.isnan(end1_val):
                print(f"NaN value found in '{adc_end0_col}' or '{adc_end1_col}' for event {row['pf_event']}, skipping")
                continue

            end0.append(end0_val)
            end1.append(end1_val)

        end0 = np.array(end0)
        end1 = np.array(end1)
        ends = [end0, end1]

        # check the pulse shape
        for end in ends:
            if spike_filter:
                # Check for spikes in data
                # TODO: change threshold to be dependent on the channel pedestal standard deviation.
                peaks, _ = sig.find_peaks((end * -1 + max(end)), prominence=1.5, threshold=10)
                if len(peaks) >= 1:

                    if index not in faulty_data_indecies:
                        faulty_data_indecies.append(index)
                        """plt.plot(end)
                        print('peaks: ', peaks)
                        plt.plot(peaks, np.array(end)[peaks.astype(int)], "x")
                        plt.show()"""

            # check for late triggers in data
            # TODO: this filters away to many good events, make stricter criteria.
            if late_trigger_filter:

                tolerance = 0.2
                if end[-1] >= (end[0] + max(end) * tolerance) or (end[-1] <= end[0] - max(end) * tolerance):

                    tolerance = 0.1
                    if end[1] <= (end[0] + end[0] * tolerance) and end[1] >= (end[0] - end[0] * tolerance):
                        if end[2] <= (end[0] + end[0] * tolerance) and end[2] >= (end[0] - end[0] * tolerance):

                            if index not in faulty_data_indecies:
                                faulty_data_indecies.append(index)

    print('number of faulty events: ', len(faulty_data_indecies))
    return faulty_data_indecies, len(faulty_data_indecies)


def select_faulty_data_new(data_df, pedestal_df, spike_filter=True, late_trigger_filter=True):
    '''
    Method for filtering out readout errors in the data
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class
    :param pedestal_df: Pandas DataFrame containing data produced by the calculatePedestals.py class
    :param spike_filter: Boolean that toggles if the function filters out spike errors
    :param late_trigger_filter: Boolean that toggles if the function filters out spike errors
    :return: faulty_data_indecies, nbr_faulty_pulses A list of the indicies of where readout errors are found in the data and
    the number of errors found
    '''
    # loop through all events in the dataframe
    print('filtering faulty events...')
    faulty_data_indecies = []
    for index, row in data_df.iterrows():

        # extract the pulse shape for the event
        end0 = []
        end1 = []

        for j in range(8):

            adc_end0_col = f'adc_{j}_end0'
            adc_end1_col = f'adc_{j}_end1'

            if adc_end0_col not in row or adc_end1_col not in row:
                print(f"Columns '{adc_end0_col}' or '{adc_end1_col}' do not exist in event {row['pf_event']}, skipping")
                continue

            end0_val = row[adc_end0_col]
            end1_val = row[adc_end1_col]

            if np.isnan(end0_val) or np.isnan(end1_val):
                print(f"NaN value found in '{adc_end0_col}' or '{adc_end1_col}' for event {row['pf_event']}, skipping")
                continue

            end0.append(end0_val)
            end1.append(end1_val)

        end0 = np.array(end0)
        end1 = np.array(end1)
        ends = [end0, end1]

        # check the pulse shape
        for end in ends:
            if spike_filter:
                # Check for spikes in data
                # TODO: change threshold to be dependent on the channel pedestal standard deviation.
                peaks, _ = sig.find_peaks((end * -1 + max(end)), prominence=1.5, threshold=10)
                if len(peaks) >= 1:

                    if index not in faulty_data_indecies:
                        faulty_data_indecies.append(index)
                        """plt.plot(end)
                        print('peaks: ', peaks)
                        plt.plot(peaks, np.array(end)[peaks.astype(int)], "x")
                        plt.show()"""

            # check for late triggers in data
            # TODO: this filters away to many good events, make stricter criteria.
            if late_trigger_filter:

                tolerance = 0.2
                if end[-1] >= (end[0] + max(end) * tolerance) or (end[-1] <= end[0] - max(end) * tolerance):

                    tolerance = 0.1
                    if end[1] <= (end[0] + end[0] * tolerance) and end[1] >= (end[0] - end[0] * tolerance):
                        if end[2] <= (end[0] + end[0] * tolerance) and end[2] >= (end[0] - end[0] * tolerance):

                            if index not in faulty_data_indecies:
                                faulty_data_indecies.append(index)

    print('number of faulty events: ', len(faulty_data_indecies))
    return faulty_data_indecies, len(faulty_data_indecies)


def subtract_pedestal(data_df, pedestal_df):
    for layer in range(1, data_df['layer'].max() + 1):

        for strip in range(data_df['strip'].max()):
            selection_data = (data_df['layer'] == layer) & (data_df['strip'] == strip)

            for end in range(2):
                selection_pedestal = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (pedestal_df['end'] == end)
                pedestal = pedestal_df[selection_pedestal]
                if pedestal.empty:
                    continue
                for data_point in range(8):
                    data_df.loc[selection_data, 'adc_' + str(data_point) + '_end' + str(end)] = data_df.loc[selection_data, 'adc_' + str(data_point) + '_end' + str(end)] - pedestal['pedestal_per_time_sample_mean'].iloc[0]

    return data_df


def select_true_MIPs(data_df, pedestal_df, signal_threshold=5, depth=3, adc_flag='adc_sum', isolated_signal_mode=False):
    '''
    Method that selects likely Minimally Ionising Particle (MIP) interactions in the data by checking for
    consecutive, high energy hits in a line through the HCal
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class
    :param pedestal_df: Pandas DataFrame containing data produced by the calculatePedestals.py class
    :param signal_threshold: the minimum distance above the pedestal (in pedestal standard deviations)
    that a readout must exceed to be classified as a signal event.
    :param depth: the search depth/number of consecutive bars in which a signal event must be detected
    for the data to be classified as a MIP event
    :param adc_flag: the metric used to gauge the size of the readout (adc_sum, adc_mean, or adc_max)
    :param isolated_signal_mode: a boolean for toggling the algorithm to only consider signal events
    in layers where there are no signal events in adjacent bars.
    :return: A Pandas DataFrame that contains data tagged as MIP interactions
    '''
    print('selecting MIPs...')
    event_max = max(data_df['pf_event'])
    true_mip_filtered = data_df.groupby(['pf_event']).apply(lambda group:
                                                            __true_MIP_event(group, pedestal_df, event_max, signal_threshold, depth, adc_flag, isolated_signal_mode))
    true_mip_filtered = true_mip_filtered.reset_index(drop=True)
    return true_mip_filtered


def __true_MIP_event(group, pedestal_df, event_max, signal_threshold, depth, adc_flag, isolated_signal_mode):
    '''
    private help function called by select_true_MIPs
    '''
    if int(group.name) % 100 == 0:
        print(int(group.name)/event_max * 100, '%')
    # selection = ((group['toa_end0'] != 0) & (group['toa_end1'] != 0))
    # data_df = group[selection]
    data_df = group
    if data_df.empty:
        return data_df
    if len(data_df['layer'].unique()) != 19:
        return pd.DataFrame()
    candidates = collections.OrderedDict()
    for layer in range(1, data_df['layer'].max() + 1):

        for strip in range(group['strip'].max() + 1):
            selection_data = (data_df['layer'] == layer) & (data_df['strip'] == strip)
            data = data_df[selection_data]
            if not data.empty:

                selection_pedestal = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip)
                pedestal = pedestal_df[selection_pedestal]
                pedestal0 = pedestal[(pedestal['end'] == 0)]
                pedestal1 = pedestal[(pedestal['end'] == 1)]
                end0_val = data[adc_flag + '_end0'].iloc[0]
                end1_val = data[adc_flag + '_end1'].iloc[0]
                if adc_flag == 'adc_sum':
                    pedestal0_val = pedestal0['pedestal'].iloc[0]
                    pedestal1_val = pedestal1['pedestal'].iloc[0]
                    pedestal0_std_dev = pedestal0['std_dev'].iloc[0]
                    pedestal1_std_dev = pedestal1['std_dev'].iloc[0]
                else:
                    pedestal0_val = pedestal0['pedestal_per_time_sample_mean'].iloc[0]
                    pedestal1_val = pedestal1['pedestal_per_time_sample_mean'].iloc[0]
                    pedestal0_std_dev = pedestal0['pedestal_per_time_sample_std_dev'].iloc[0]
                    pedestal1_std_dev = pedestal1['pedestal_per_time_sample_std_dev'].iloc[0]
                if (end0_val > pedestal0_val + pedestal0_std_dev * signal_threshold) and (end1_val > pedestal1_val + pedestal1_std_dev * signal_threshold):
                    candidates[str(layer) + '-' + str(strip)] = True
                else:
                    candidates[str(layer) + '-' + str(strip)] = False
            else:
                candidates[str(layer) + '-' + str(strip)] = False
    true_mips = __line_up_event(candidates, group, depth, isolated_signal_mode)
    true_mips = pd.DataFrame(true_mips, columns=['layer', 'strip'])
    filtered_group = pd.merge(group, true_mips, on=['layer', 'strip'], how='inner')
    return filtered_group


def __line_up_event(candidates, group, depth, isolated_signal_mode):
    '''
    private help function called by __true_MIP_event
    '''
    true_mip_list = []
    if depth > 3:
        side_depth = 3
    else:
        side_depth = depth
    # loop through all channels in event
    for layer in range(1, 20):

        # handle that different parts of detector have different number of layers
        if layer < 10:
            nbr_bars = 8
        else:
            nbr_bars = 12

        # check how many layers of same orientation is in front and behind the current layer
        nbr_layers_infront = math.floor((layer - 1) / 2)
        nbr_layers_behind = math.floor((19 - layer) / 2)

        # determine how many layers in front and behind current layer to check
        temp_front_check = math.floor(depth/2)
        temp_back_check = math.ceil(depth/2)

        back_weight = temp_back_check - nbr_layers_behind
        front_weight = temp_front_check - nbr_layers_infront

        if back_weight < 0:
            back_weight = 0
        if front_weight < 0:
            front_weight = 0

        back_check = temp_back_check + front_weight - back_weight
        front_check = temp_front_check + back_weight - front_weight

        # The back part of the detector has extra bars on the side that needs to be handled
        if layer > 9:

            nbr_layers_infront_side = math.floor((layer - 9) / 2)

            temp_front_side_check = math.floor(side_depth / 2)
            temp_back_side_chack = math.ceil(side_depth / 2)

            front_side_weight = temp_front_side_check - nbr_layers_infront_side
            back_side_weight = temp_back_side_chack - nbr_layers_behind

            if front_side_weight < 0:
                front_side_weight = 0
            if back_side_weight < 0:
                back_side_weight = 0

            front_side_check = temp_front_side_check + back_side_weight - front_side_weight
            back_side_check = temp_back_side_chack + front_side_weight - back_side_weight

        else:

            back_side_check = 0
            front_side_check = 0
            nbr_layers_infront_side = 0

        if(back_check > nbr_layers_behind or front_check > nbr_layers_infront):
            print('Warning: depth to large to accommodate in detector geometry, ignoring')
            return []

        # front half of detector
        if layer <= 9:
            # check every bar in layer if it lines up with a MIP line
            for bar in range(nbr_bars):
                true_mip = False
                falsified = False
                #check in front of current bar
                for i in range(front_check):
                    check_layer = layer - 2 - 2 * i
                    check_bar = bar
                    if candidates[str(check_layer) + '-' + str(check_bar)]:
                        true_mip = True
                    else:
                        true_mip = False
                        falsified = True
                        break
                if not falsified:
                    # check behind current bar
                    for i in range(back_check):
                        check_layer = layer + 2 + 2 * i
                        if check_layer > 9:
                            check_bar = bar + 2
                        else:
                            check_bar = bar
                        if candidates[str(check_layer) + '-' + str(check_bar)]:
                            true_mip = True
                        else:
                            true_mip = False
                            break
                # check adjacent bars for signal if in isolated_signal_mode
                if isolated_signal_mode:
                    check_layer = layer
                    check_bar_left = bar - 1
                    check_bar_right = bar + 1
                    if candidates[str(check_layer) + '-' + str(check_bar_left)] or candidates[
                        str(check_layer) + '-' + str(check_bar_right)]:
                        true_mip = False
                if true_mip:
                    true_mip_list.append(tuple([layer, bar]))

        # back half of detector
        else:
            # check every bar in layer if it lines up with a MIP line
            for bar in range(nbr_bars):
                true_mip = False
                falsified = False
                # side bar case
                if bar < 2 or bar >= 10:
                    # check in front of current bar
                    for i in range(front_side_check):
                        check_layer = layer - 2 - 2 * i
                        check_bar = bar
                        if candidates[str(check_layer) + '-' + str(check_bar)]:
                            true_mip = True
                        else:
                            true_mip = False
                            falsified = True
                            break
                    if not falsified:
                        # check behind current bar
                        for i in range(back_side_check):
                            check_layer = layer + 2 + 2 * i
                            check_bar = bar
                            if candidates[str(check_layer) + '-' + str(check_bar)]:
                                true_mip = True
                            else:
                                true_mip = False
                                break
                    # check adjacent bars for signal if in isolated_signal_mode
                    if isolated_signal_mode:
                        check_layer = layer
                        check_bar_left = bar - 1
                        check_bar_right = bar + 1
                        if candidates[str(check_layer) + '-' + str(check_bar_left)] or candidates[
                            str(check_layer) + '-' + str(check_bar_right)]:
                            true_mip = False
                # center bar case
                else:
                    # check every bar in layer if it lines up with a MIP line
                    # check in front of current bar
                    for i in range(front_check):
                        check_layer = layer - 2 - 2 * i
                        if check_layer <= 9:
                            check_bar = bar - 2
                        else:
                            check_bar = bar
                        if candidates[str(check_layer) + '-' + str(check_bar)]:
                            true_mip = True
                        else:
                            true_mip = False
                            falsified = True
                            break
                    if not falsified:
                        # check behind current bar
                        for i in range(back_check):
                            check_layer = layer + 2 + 2 * i
                            check_bar = bar
                            if candidates[str(check_layer) + '-' + str(check_bar)]:
                                true_mip = True
                            else:
                                true_mip = False
                                break
                    # check adjacent bars for signal if in isolated_signal_mode
                    if isolated_signal_mode:
                        check_layer = layer
                        check_bar_left = bar - 1
                        check_bar_right = bar + 1
                        if candidates[str(check_layer) + '-' + str(check_bar_left)] or candidates[
                            str(check_layer) + '-' + str(check_bar_right)]:
                            true_mip = False
                if true_mip:
                    true_mip_list.append(tuple([layer, bar]))
    return  true_mip_list


def __line_up_events_old(candidates, group):
    '''
    NOTE!! OUTDATED!!
    '''
    true_mip = []
    longest_count = 0
    length_threshold = 4

    # loop through all seeds and try to line up with other candidates
    for seed in candidates:
        current_mip = []
        count = 1

        # check if we're in an even layer or odd layer
        if seed[0] % 2 == 0:

            # loop through even layers
            for layer in range(2, group['layer'].max() + 1, 2):

                # check if we're in the front or back part of the detector and check if the seed lines up with other
                # candidates
                if (layer <= 9 and seed[0] <= 9) or (layer >= 10 and seed[0] >= 10):
                    if tuple([layer, seed[1]]) in candidates:
                        current_mip.append(tuple([layer, seed[1]]))
                        count += 1
                elif layer >= 9 and seed[0] <= 9:
                    if tuple([layer, seed[1] + 2]) in candidates:
                        current_mip.append(tuple([layer, seed[1]]))
                        count += 1
                elif layer <= 9 and seed[0] >= 9:
                    if tuple([layer, seed[1] - 2]) in candidates:
                        current_mip.append(tuple([layer, seed[1] - 2]))
                        count += 1

        # for odd layers
        else:
            for layer in range(1, group['layer'].max() + 1, 2):

                # check if we're in the front or back part of the detector on the current candidate and for the seed.
                # Check if the seed lines up with the current candidate.
                if (layer <= 9 and seed[0] <= 9) or (layer >= 10 and seed[0] >= 10):
                    if tuple([layer, seed[1]]) in candidates:
                        current_mip.append(tuple([layer, seed[1]]))
                        count += 1
                elif layer >= 9 and seed[0] <= 9:
                    if tuple([layer, seed[1] + 2]) in candidates:
                        current_mip.append(tuple([layer, seed[1]]))
                        count += 1
                elif layer <= 9 and seed[0] >= 9:
                    if tuple([layer, seed[1] - 2]) in candidates:
                        current_mip.append(tuple([layer, seed[1] - 2]))
                        count += 1

        # check if the current mip is the longest track we've found
        if count > longest_count and count > length_threshold:
            longest_count = count
            true_mip = current_mip.copy()
    return true_mip


def event_fraction_channelwise(data_df1, data_df2):
    result_df = pd.DataFrame()
    for layer in range(1, data_df1['layer'].max() + 1):

        for strip in range(data_df1['strip'].max()):

            for end in range(2):
                selection_data1 = (data_df1['layer'] == layer) & (data_df1['strip'] == strip)
                selection_data2 = (data_df2['layer'] == layer) & (data_df2['strip'] == strip)
                data1 = data_df1[selection_data1]
                data2 = data_df2[selection_data2]
                try:
                    fraction = data2.size / data1.size
                except ZeroDivisionError:
                    fraction = np.nan
                temp = pd.DataFrame([{'layer': layer, 'strip': strip, 'end': end, 'fraction': fraction}])
                result_df = pd.concat([result_df, temp])
    return result_df


def calculate_MIP_efficiency(tagged_MIPs_df, pedestal_df, adc_flag='adc_mean', show_plot=True, save_directory=None, two_sided=False):
    '''
    Calculates the MIP detection efficiency at 5 pedestal standard deviations above the pedestal from
    tagged MIP events and Pedestal data then plots it as a function of distance above the pedestal
    :param tagged_MIPs_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class,
    and further selected for MIPs by select_true_MIPs
    :param pedestal_df: Pandas DataFrame containing data produced by the calculatePedestals.py class
    :param adc_flag: the metric used to gauge the size of the readout (adc_sum, adc_mean, or adc_max)
    :param show_plot: Boolean that toggles if the plots are shown during runtime
    :param save_directory: directory address in which to save the generated plots
    :param two_sided: Boolean that toggles if the MIP efficiency is calculated for the two ends
    of the scintillator bars individually (False) or jointly for a full bar (True)
    :return: Pandas DataFrame that contains the MIP efficiency 5 standard deviation above the
    pedestal for all channels or bars.
    '''
    # Calculates the MIP efficiency of a data frame that has been selected for MIPs
    # TODO: include threshold in pedestal sigma as parameter
    print('calculating MIP efficiencies...')
    result_df = pd.DataFrame()
    for layer in range(1, int(tagged_MIPs_df['layer'].max() + 1)):

        for strip in range(int(tagged_MIPs_df['strip'].max() + 1)):
            selection_data = (tagged_MIPs_df['layer'] == layer) & (tagged_MIPs_df['strip'] == strip)
            data = tagged_MIPs_df[selection_data]

            if not two_sided:

                for end in range(2):
                    selection_pedestal = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (pedestal_df['end'] == end)
                    pedestal = pedestal_df[selection_pedestal]
                    data_vals = data[adc_flag + '_end' + str(end)].to_numpy()
                    if len(data_vals) == 0:
                        continue
                    pedestal_val = pedestal['pedestal_per_time_sample_mean'].iloc[0]
                    pedestal_std_dev = pedestal['pedestal_per_time_sample_std_dev'].iloc[0]
                    if adc_flag == 'adc_max':
                        adc_max_threshold = pedestal['adc_max_threshold'].iloc[0]

                    # try catch fix for a stupid bug in matplotlib where the plot sometimes randomly does not work
                    done = False
                    while not done:
                        try:
                            label = 'fraction of MIPs detected'

                            n, bins, patches = plt.hist(data_vals, bins=1024, range=[0, 1024],
                                                         density=True, cumulative=-1, histtype='step',
                                                         label=label)

                            if adc_flag == 'adc_max':
                                plt.vlines(adc_max_threshold, 0, 1.5, label=(r'false rate from pedestal < $10^{-6}$'), color='red')

                            else:
                                plt.vlines(pedestal_val, 0, 1.5, label=(r'pedestal $\pm$ 5' + r'$\sigma$'), color='red')
                                plt.axvspan(pedestal_val - 5 * pedestal_std_dev, pedestal_val + 5 * pedestal_std_dev, alpha=0.3, color='red')
                            plt.title('Test beam MIP efficiency layer ' + str(layer) + ' bar ' + str(strip) +
                                      ' end ' + str(end))
                            plt.xlim(0, 400)
                            plt.ylim(0, 1.5)
                            if adc_flag == 'adc_max':
                                plt.xlabel('Detection threshold [max ADC]')
                            elif adc_flag == 'adc_sum':
                                plt.xlabel('Detection threshold [sum ADC]')
                            elif adc_flag == 'adc_mean':
                                plt.xlabel('Detection threshold [mean ADC]')
                            plt.ylabel('fraction of MIPs detected')
                            plt.legend()
                            if show_plot:
                                plt.show()
                            if save_directory is not None:
                                plt.savefig(save_directory + "/True_MIP_Efficiency_" + adc_flag +"_layer_"
                                            + str(layer) + "_bar_" + str(strip) + "_end_" + str(end) + ".png")
                            plt.close()
                            done = True
                        except ValueError:
                            continue
                    if adc_flag == 'adc_max':
                        efficiency = {'MIP_efficiency': [n[math.ceil(adc_max_threshold)]],
                                      'layer': [layer], 'strip': [strip], 'end': [end]}

                    else:
                        efficiency = {'MIP_efficiency': [n[math.ceil(pedestal_val + pedestal_std_dev * 5)]],
                                      'layer': [layer], 'strip': [strip], 'end': [end]}

                    efficiency_df = pd.DataFrame(efficiency)
                    result_df = pd.concat([result_df, efficiency_df])

            else:
                # Two sided efficiency case
                # select pedestal and data values
                selection_pedestal_0 = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (
                            pedestal_df['end'] == 0)
                pedestal_0 = pedestal_df[selection_pedestal_0]
                data_vals_0 = data[adc_flag + '_end' + str(0)].to_numpy()

                selection_pedestal_1 = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (
                        pedestal_df['end'] == 1)
                pedestal_1 = pedestal_df[selection_pedestal_1]
                data_vals_1 = data[adc_flag + '_end' + str(1)].to_numpy()

                if len(data_vals_0) == 0 or len(data_vals_1) == 0:
                    continue

                pedestal_val_0 = pedestal_0['pedestal_per_time_sample_mean'].iloc[0]
                pedestal_std_dev_0 = pedestal_0['pedestal_per_time_sample_std_dev'].iloc[0]
                pedestal_val_1 = pedestal_1['pedestal_per_time_sample_mean'].iloc[0]
                pedestal_std_dev_1 = pedestal_1['pedestal_per_time_sample_std_dev'].iloc[0]

                # Translate data values from ADC units to distance from pedestal in pedestal standard deviation units
                data_vals_0 = (data_vals_0 - pedestal_val_0) / pedestal_std_dev_0
                data_vals_1 = (data_vals_1 - pedestal_val_1) / pedestal_std_dev_1

                if adc_flag == 'adc_max':
                    adc_max_threshold_1 = pedestal_1['adc_max_threshold'].iloc[0]
                    adc_max_threshold_0 = pedestal_0['adc_max_threshold'].iloc[0]
                    adc_max_threshold_1 = (adc_max_threshold_1 - pedestal_val_1) / pedestal_std_dev_1
                    adc_max_threshold_0 = (adc_max_threshold_0 - pedestal_val_0) / pedestal_std_dev_0
                    # These should be the same since they are defined as a set number of standard deviations, but just in case:
                    two_sided_adc_max_threshold = np.maximum(adc_max_threshold_0, adc_max_threshold_1)

                # Merge the two ends such that only the largest signal in the bar remains
                two_sided_data = np.maximum(data_vals_0, data_vals_1)

                # Calculate efficiency and plot
                # Fix for dumb matplotlib bug
                done = False
                while not done:
                    try:
                        label = 'fraction of MIPs detected'

                        n, bins, patches = plt.hist(two_sided_data, bins=10000, range=[0, 1000],
                                                    density=True, cumulative=-1, histtype='step',
                                                    label=label)

                        if adc_flag == 'adc_max':
                            plt.vlines(two_sided_adc_max_threshold, 0, 1.5, label=(r'false rate from pedestal < $10^{-6}$'),
                                       color='red')

                        else:
                            plt.vlines(5, 0, 1.5, label=(r'pedestal + 5' + r'$\sigma$'), color='red')
                        plt.title('Test beam two sided MIP efficiency on ' + adc_flag + ' layer ' + str(layer) + ' bar ' + str(strip))
                        plt.xlim(0, 200)
                        plt.ylim(0, 1.5)
                        plt.xlabel(r'Detection threshold [Pedestal $\sigma$]')
                        plt.ylabel('fraction of MIPs detected')
                        plt.legend()
                        if show_plot:
                            plt.show()
                        if save_directory is not None:
                            plt.savefig(save_directory + "/Two_sided_MIP_Efficiency_" + adc_flag + "_layer_"
                                        + str(layer) + "_bar_" + str(strip) + ".png")
                        plt.close()
                        done = True
                    except ValueError:
                        continue
                if adc_flag == 'adc_max':
                    efficiency = {'Two_sided_MIP_efficiency': [n[np.where((two_sided_adc_max_threshold - 0.1 <= bins) | (bins <= two_sided_adc_max_threshold + 0.1))[0][0]]],
                                  'layer': [layer], 'strip': [strip]}

                else:
                    efficiency = {'Two_sided_MIP_efficiency': [n[np.where(bins == 5)[0][0]]],
                                  'layer': [layer], 'strip': [strip]}

                efficiency_df = pd.DataFrame(efficiency)
                result_df = pd.concat([result_df, efficiency_df])

    return result_df


def makeLabel():
    hep.cms.text(exp="Experiment", text="Internal", fontsize=11, loc=0)


def plot_MIP_efficiency_comparison(data_df, MC_df, pedestal_df, plot_dir, adc_flag='adc_mean', show_plot=True, save_plot=False, ped_sub=False, two_sided=False, export_hist_to_csv=False):
    '''
    plots a comparison of MIP efficiency between two data sets assumed to be experimental and Monte Carlo data
    :param data_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class,
    and further selected for MIPs by select_true_MIPs
    :param MC_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class,
    and further selected for MIPs by select_true_MIPs
    :param pedestal_df: Pandas DataFrame containing data produced by the calculatePedestals.py class
    :param plot_dir: directory address in which to save the generated plots and data
    :param adc_flag: the metric used to gauge the size of the readout (adc_sum, adc_mean, or adc_max)
    :param show_plot: Boolean that toggles if the plots are shown during runtime
    :param save_plot: Boolean that toggles if the plots are saved
    :param ped_sub: Boolean that toggles if the pedestals are subtracted in the histogram
    :param two_sided: Boolean that toggles if the MIP efficiency is calculated for the two ends
    of the scintillator bars individually (False) or jointly for a full bar (True)
    :param export_hist_to_csv: Boolean that toggles if the data is saved to csv file
    '''
    # Some plotting setup
    plt.get_backend()
    plt.style.use(hep.style.ROOT)
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

    # Plotting the MIP efficiency of two data frames that have been selected for MIPs
    print('plotting MIP efficiencies...')

    for layer in range(1, int(data_df['layer'].max() + 1)):

        for strip in range(int(data_df['strip'].max() + 1)):

            selection_data1 = (data_df['layer'] == layer) & (data_df['strip'] == strip)
            data = data_df[selection_data1]

            selection_data2 = (MC_df['layer'] == layer) & (MC_df['strip'] == strip)
            MC = MC_df[selection_data2]

            if not two_sided:

                for end in range(2):
                    selection_pedestal = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (pedestal_df['end'] == end)
                    pedestal = pedestal_df[selection_pedestal]
                    data_vals = data[adc_flag + '_end' + str(end)].to_numpy()
                    MC_vals = MC[adc_flag + '_end' + str(end)].to_numpy()

                    if len(data_vals) == 0 or len(MC_vals) == 0:
                        continue
                    pedestal_val = pedestal['pedestal_per_time_sample_mean'].iloc[0]
                    pedestal_std_dev = pedestal['pedestal_per_time_sample_std_dev'].iloc[0]

                    data_vals_pedestal_subtracted = data_vals - pedestal_val
                    MC_vals_pedestal_subtracted = MC_vals - pedestal_val

                    # try catch fix for a stupid bug in matplotlib where the plot sometimes randomly does not work
                    done = False
                    while not done:
                        try:
                            label1 = 'Fraction of MIPs detected [test beam]'
                            label2 = 'Fraction of MIPs detected [Monte Carlo]'

                            if ped_sub:
                                n1, bins1, patches1 = plt.hist(data_vals_pedestal_subtracted, bins=1024, range=[0, 1024],
                                                               density=True, cumulative=-1, histtype='step',
                                                               label=label1, color='blue')
                                n2, bins2, patches2 = plt.hist(MC_vals_pedestal_subtracted, bins=1024, range=[0, 1024],
                                                               density=True, cumulative=-1, histtype='step',
                                                               label=label2, color='black')
                                plt.vlines(5 * pedestal_std_dev, 0, 1.5, label=(r'pedestal + 5$\sigma$'), color='red')

                                plt.title('MIP efficiencies layer ' + str(layer) + ' bar ' + str(strip) +
                                          ' end ' + str(end))
                                plt.xlim(0, 300)
                                plt.ylim(0, 1.5)
                                plt.xlabel('Detection threshold [mean ADC above pedestal]')
                                plt.ylabel('Fraction of MIPs detected')

                            else:
                                n1, bins1, patches1 = plt.hist(data_vals, bins=1024, range=[0, 1024],
                                                             density=True, cumulative=-1, histtype='step',
                                                             label=label1, color='blue')
                                n2, bins2, patches2 = plt.hist(MC_vals, bins=1024, range=[0, 1024],
                                                            density=True, cumulative=-1, histtype='step',
                                                            label=label2, color='black')
                                plt.vlines(pedestal_val, 0, 1.5, label=(r'pedestal $\pm$ 5' + r'$\sigma$'), color='red')
                                plt.axvspan(pedestal_val - 5 * pedestal_std_dev, pedestal_val + 5 * pedestal_std_dev, alpha=0.3, color='red')

                                plt.title('MIP efficiencies layer ' + str(layer) + ' bar ' + str(strip) +
                                          ' end ' + str(end))
                                plt.xlim(0, 400)
                                plt.ylim(0, 1.5)
                                plt.xlabel('Detection threshold [mean ADC]')
                                plt.ylabel('Fraction of MIPs detected')

                            plt.legend()
                            if show_plot:
                                plt.show()
                            if save_plot:
                                plt.savefig(plot_dir + "/MIP_Efficiency_comparison_layer_" + str(layer) + "_bar_" + str(strip) + "_end_" + str(end) + ".png")
                            plt.close()
                            done = True
                        except ValueError:
                            continue

            else:
                # Two sided case
                selection_pedestal_0 = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (
                        pedestal_df['end'] == 0)
                pedestal_0 = pedestal_df[selection_pedestal_0]
                data_vals_0 = data[adc_flag + '_end' + str(0)].to_numpy()
                MC_vals_0 = MC[adc_flag + '_end' + str(0)].to_numpy()

                selection_pedestal_1 = (pedestal_df['layer'] == layer) & (pedestal_df['strip'] == strip) & (
                        pedestal_df['end'] == 1)
                pedestal_1 = pedestal_df[selection_pedestal_1]
                data_vals_1 = data[adc_flag + '_end' + str(1)].to_numpy()
                MC_vals_1 = MC[adc_flag + '_end' + str(1)].to_numpy()

                if len(data_vals_0) == 0 or len(data_vals_1) == 0 or len(MC_vals_0) == 0 or len(MC_vals_1) == 0:
                    continue

                pedestal_val_0 = pedestal_0['pedestal_per_time_sample_mean'].iloc[0]
                pedestal_std_dev_0 = pedestal_0['pedestal_per_time_sample_std_dev'].iloc[0]
                pedestal_val_1 = pedestal_1['pedestal_per_time_sample_mean'].iloc[0]
                pedestal_std_dev_1 = pedestal_1['pedestal_per_time_sample_std_dev'].iloc[0]

                # Translate data values from ADC units to distance from pedestal in pedestal standard deviation units
                data_vals_0 = (data_vals_0 - pedestal_val_0) / pedestal_std_dev_0
                data_vals_1 = (data_vals_1 - pedestal_val_1) / pedestal_std_dev_1

                MC_vals_0 = (MC_vals_0 - pedestal_val_0) / pedestal_std_dev_0
                MC_vals_1 = (MC_vals_1 - pedestal_val_1) / pedestal_std_dev_1

                if adc_flag == 'adc_max':
                    adc_max_threshold_1 = pedestal_1['adc_max_threshold'].iloc[0]
                    adc_max_threshold_0 = pedestal_0['adc_max_threshold'].iloc[0]
                    adc_max_threshold_1 = (adc_max_threshold_1 - pedestal_val_1) / pedestal_std_dev_1
                    adc_max_threshold_0 = (adc_max_threshold_0 - pedestal_val_0) / pedestal_std_dev_0
                    # These should be the same since they are defined as a set number of standard deviations, but just in case:
                    two_sided_adc_max_threshold = np.maximum(adc_max_threshold_0, adc_max_threshold_1)

                # Merge the two ends such that only the largest signal in the bar remains
                two_sided_data = np.maximum(data_vals_0, data_vals_1)
                two_sided_MC = np.maximum(MC_vals_0, MC_vals_1)

                # try catch fix for a stupid bug in matplotlib where the plot sometimes randomly does not work
                done = False
                while not done:
                    try:
                        fig = plt.gcf()
                        fig.set_size_inches(figureWidth, 2)
                        label1 = 'Fraction of MIPs detected [test beam]'
                        label2 = 'Fraction of MIPs detected [Monte Carlo]'
                        n, bins, patches = plt.hist(two_sided_data, bins=10000, range=[0, 1000],
                                                    density=True, cumulative=-1, histtype = 'step',
                                                    color = dataColor, label = "Data")
                        n2, bins2, patches2 = plt.hist(two_sided_MC, bins=10000, range=[0, 1000],
                                                       density=True, cumulative=-1, histtype='step',
                                                    color=mcColor, label="MC", linestyle="--")

                        if export_hist_to_csv:
                            n = np.append(n, np.nan)
                            n2 = np.append(n2, np.nan)
                            hist_dict = {'bins': bins, 'data': n, 'MC': n2}
                            hist_df = pd.DataFrame(hist_dict)
                            hist_df.to_csv(
                                plot_dir + "//two_sided_MIP_Efficiency_comparison_layer_" + str(layer) + "_bar_" + str(
                                strip) + ".csv")


                        if adc_flag == 'adc_max':
                            plt.vlines(two_sided_adc_max_threshold, 0, 1.5, label=(r'false rate from pedestal < $10^{-6}$'),
                                       color='red')
                        else:
                            plt.vlines(5, 0, 1.5, label=r"Pedestal + 5$\sigma$", color=oddColor2)

                        plt.xlim(0, 150)
                        plt.ylim(0, 1.2)
                        plt.xlabel(r"Readout Threshold [$\sigma$ Above Pedestal]")
                        plt.ylabel("MIP Detection Efficiency")
                        makeLabel()
                        plt.legend(loc=1)

                        plt.legend()
                        if show_plot:
                            plt.show()
                        if save_plot:
                            plt.savefig(plot_dir + "/two_sided_MIP_Efficiency_comparison_layer_" + str(layer) + "_bar_" + str(
                                strip) + ".png", bbox_inches = "tight")
                        plt.close()
                        done = True
                    except ValueError:
                        continue


def plot_pulse_shapes(pulses_df):

    nbr_events = len(pulses_df)
    fig, axes = plt.subplots(max(2, nbr_events) , 2, figsize=(10, 5 * nbr_events))
    for i in range(nbr_events):
        event = pulses_df.iloc[[i]]
        if event.empty:
            print(f"Event {i} is empty, skipping")
            continue

        end0 = []
        end1 = []
        x = []
        for j in range(8):
            adc_end0_col = f'adc_{j}_end0'
            adc_end1_col = f'adc_{j}_end1'

            if adc_end0_col not in event or adc_end1_col not in event:
                print(f"Columns '{adc_end0_col}' or '{adc_end1_col}' do not exist in event {i}, skipping")
                continue

            end0_val = event[adc_end0_col].values[0]
            end1_val = event[adc_end1_col].values[0]

            if np.isnan(end0_val) or np.isnan(end1_val):
                print(f"NaN value found in '{adc_end0_col}' or '{adc_end1_col}' for event {i}, skipping")
                continue

            end0.append(end0_val)
            end1.append(end1_val)
            x.append(j + 1)

        if not end0 or not end1:
            print(f"No valid data for event {i}, skipping plot")
            continue

        print('end0: ', end0)
        print('end1: ', end1)

        try:
            axes[i][0].plot(x, end0)
            axes[i][1].plot(x, end1)
            axes[i][0].set_ylim([0, 250])
            axes[i][1].set_ylim([0, 250])
            axes[i][0].set_xlabel('time (ordered datapoint)')
            axes[i][1].set_xlabel('time (ordered datapoint)')
            axes[i][0].set_ylabel('adc')
            axes[i][1].set_ylabel('adc')
            # axes[i][0].set_title('event ' + str(event['pf_event'].values[0]) + ' end 0')
            # axes[i][1].set_title('event ' + str(event['pf_event'].values[0]) + ' end 1')
            axes[i][0].set_title('Layer ' + str(event['layer'].iloc[0]) + ' bar ' + str(event['strip'].iloc[0]) + ' end 0')
            axes[i][1].set_title('Layer ' + str(event['layer'].iloc[0]) + ' bar ' + str(event['strip'].iloc[0]) + ' end 1')
        except Exception as e:
            print(f"Error plotting event {i}: {e}")

    fig.suptitle('Pulse shape anomaly correlation')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # fig.suptitle('Layer ' + str(event['layer']) + ' bar ' + str(event['strip']))
    # plt.show()


def get_corr_hist(faulty_df, normalise=None):
    '''
    Method that generates a correlation histogram of readout errors over the detector
    :param faulty_df: Pandas DataFrame containing data produced by the makeAnalysisFiles.py class,
    and further selected to contain only the data tagged as faulty by select_faulty_data_new
    :param normalise: OUTDATED! flag that may be set to 'background' to get a background subtraction
    :return: 2-dimensional numpy array containing the correlation histogram
    '''
    print('Calculating correlations...')
    nbr_events = max(faulty_df['pf_event'])
    hist = np.zeros([192, 192])
    faulty_df.groupby('pf_event').apply(lambda group: __group_corr(group, hist, nbr_events))
    if normalise == 'background':
        __background_subtract_corr_hist(hist, faulty_df)
    print('Finished alculating correlations')
    return hist


def __group_corr(group, hist, nbr_events):
    '''
    Private help method called by get_corr_hist
    '''
    # TODO: this counts errors in both ends of the same bar as two seperate errors
    event_number = group['pf_event'].iloc[0]
    if int(event_number) % 1000 == 0:
        print(event_number/nbr_events * 100, '%')
    for index1, row1 in group.iterrows():
        for index2, row2 in group.iterrows():
            if index1 >= index2:
                continue
            layer_1 = row1['layer']
            layer_2 = row2['layer']
            bar_1 = row1['strip']
            bar_2 = row2['strip']

            if layer_1 <= 9 and layer_2 <= 9:
                hist[int((layer_1 - 1) * 8 + bar_1), int((layer_2 - 1) * 8 + bar_2)] += 1
                hist[int((layer_2 - 1) * 8 + bar_2), int((layer_1 - 1) * 8 + bar_1)] += 1

            elif layer_1 <= 9 and layer_2 > 9:
                hist[int((layer_1 - 1) * 8 + bar_1), int(9 * 8 + (layer_2 - 10) * 12 + bar_2)] += 1
                hist[int(9 * 8 + (layer_2 - 10) * 12 + bar_2), int((layer_1 - 1) * 8 + bar_1)] += 1

            elif layer_2 <= 9 and layer_1 > 9:
                hist[int((layer_2 - 1) * 8 + bar_2), int(9 * 8 + (layer_1 - 10) * 12 + bar_1)] += 1
                hist[int(9 * 8 + (layer_1 - 10) * 12 + bar_1), int((layer_2 - 1) * 8 + bar_2)] += 1

            elif layer_2 > 9 and layer_1 > 9:
                hist[int(9 * 8 + (layer_2 - 10) * 12 + bar_2), int(9 * 8 + (layer_1 - 10) * 12 + bar_1)] += 1
                hist[int(9 * 8 + (layer_1 - 10) * 12 + bar_1), int(9 * 8 + (layer_2 - 10) * 12 + bar_2)] += 1

            """hist[int((layer_1 - 1) * 12 + bar_1), int((layer_2 - 1) * 12 + bar_2)] += 1
            hist[int((layer_2 - 1) * 12 + bar_2), int((layer_1 - 1) * 12 + bar_1)] += 1"""


def __background_subtract_corr_hist(hist, faulty_df):
    '''
    Private help method called by get_corr_hist
    '''
    total_errors = len(faulty_df.index)
    norm_array = np.zeros(192)
    for layer in range(1, 20):
        if layer <= 9:
            for bar in range(8):
                df_filter = (faulty_df['layer'] == layer) & (faulty_df['strip'] == bar)
                nbr_erors = len(faulty_df[df_filter].index)
                norm_array[int((layer - 1) * 8 + bar)] = nbr_erors

        else:
            for bar in range(12):
                df_filter = (faulty_df['layer'] == layer) & (faulty_df['strip'] == bar)
                nbr_erors = len(faulty_df[df_filter].index)
                norm_array[int(9 * 8 + (layer - 10) * 12 + bar)] = nbr_erors

    for row in range(192):
        for column in range(192):
            background_probability = (norm_array[row] / total_errors) * (norm_array[column]/total_errors)
            background = background_probability * total_errors
            hist[row][column] -= background

    return hist


def __normalise_corr_hist(hist, faulty_df):
    '''
    OUTDATED!
    '''
    total_errors = len(faulty_df.index)
    norm_array = np.zeros(192)
    for layer in range(1, 20):
        if layer <= 9:
            for bar in range(8):
                df_filter = (faulty_df['layer'] == layer) & (faulty_df['strip'] == bar)
                nbr_erors = len(faulty_df[df_filter].index)
                norm_array[int((layer - 1) * 8 + bar)] = nbr_erors

        else:
            for bar in range(12):
                df_filter = (faulty_df['layer'] == layer) & (faulty_df['strip'] == bar)
                nbr_erors = len(faulty_df[df_filter].index)
                norm_array[int(9 * 8 + (layer - 10) * 12 + bar)] = nbr_erors

    for row in range(192):
        for column in range(192):
            background_probability = (norm_array[row] / total_errors) * (norm_array[column]/total_errors)
            background = background_probability * total_errors
            hist[row][column] -= background

    return hist


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


def select_signal_on_pedestal(data_df, pedestal_df, tolerance):
    print('selecting signal events on pedestal...')
    # TODO: not robust if data_df and pedestal_df does not contain the same layer-strip combinations
    result_df = pd.DataFrame()

    for layer in range(1, pedestal_df['layer'].max() + 1):
        layer_selection = pedestal_df['layer'] == layer
        layer_pedestal = pedestal_df[layer_selection]

        for strip in range(layer_pedestal['strip'].max() + 1):
            data_strip_selection = (data_df['layer'] == layer) & (data_df['strip'] == strip)
            data_strip = data_df[data_strip_selection]

            pedestal_selection_end0 = (layer_pedestal['strip'] == strip) & (layer_pedestal['end'] == 0)
            channel_pedestal_end0 = layer_pedestal[pedestal_selection_end0]
            pedestal_value_end0 = channel_pedestal_end0['pedestal_per_time_sample_mean'].iloc[0]
            pedestal_std_dev_end0 = channel_pedestal_end0['pedestal_per_time_sample_std_dev'].iloc[0]

            pedestal_selection_end1 = (layer_pedestal['strip'] == strip) & (layer_pedestal['end'] == 1)
            channel_pedestal_end1 = layer_pedestal[pedestal_selection_end1]
            pedestal_value_end1 = channel_pedestal_end1['pedestal_per_time_sample_mean'].iloc[0]
            pedestal_std_dev_end1 = channel_pedestal_end1['pedestal_per_time_sample_std_dev'].iloc[0]

            # Select data above the pedestals in given channel
            signal_filter = (data_strip['adc_mean_end0'] > pedestal_value_end0 + tolerance * pedestal_std_dev_end0) & (data_strip['adc_mean_end1'] > pedestal_value_end1 + tolerance * pedestal_std_dev_end1)
            channel_signal_events = data_strip[signal_filter]
            result_df = pd.concat([result_df, channel_signal_events])
    print('signal calculation finished')
    return result_df


def check_signal_bleed_neighbour(MIP_df, signal_df):
    signal_df = signal_df.set_index(['pf_event', 'layer', 'strip'])
    leak_candidates_df = pd.concat(
        MIP_df.apply(lambda row: __check_neighboring_bars(row, signal_df), axis=1).dropna().tolist(), ignore_index=True)
    return leak_candidates_df.reset_index()


def check_signal_bleed_neighbor_within_quadbar(MIP_df, signal_df):
    signal_df = signal_df.set_index(['pf_event', 'layer', 'strip'])
    bar_filter = MIP_df['strip'].isin([1, 2, 5, 6, 9, 10])
    MIP_df = MIP_df[bar_filter]
    leak_candidates_df = pd.concat(MIP_df.apply(lambda row: __check_neighboring_bars(row, signal_df), axis=1).dropna().tolist(), ignore_index=True)
    return leak_candidates_df.reset_index()


def check_signal_bleed_neighbor_between_quadbars(MIP_df, signal_df):
    signal_df = signal_df.set_index(['pf_event', 'layer', 'strip'])
    bar_filter = MIP_df['strip'].isin([3, 4, 7, 8])
    MIP_df = MIP_df[bar_filter]
    leak_candidates_df = pd.concat(MIP_df.apply(lambda row: __check_neighboring_bars(row, signal_df), axis=1).dropna().tolist(), ignore_index=True)
    return leak_candidates_df.reset_index()


def __check_neighboring_bars(row, signal_df):
    candidates = []
    for neighbor_strip in [row['strip'] - 1, row['strip'] + 1]:
        try:
            key = (row['pf_event'], row['layer'], neighbor_strip)
            leak_candidate = signal_df.loc[[key]].reset_index()
        except KeyError:
            continue

        filter = (leak_candidate['adc_sum_end0'] < row['adc_sum_end0']) & (leak_candidate['adc_sum_end1'] < row['adc_sum_end1'])
        leak_candidate = leak_candidate[filter]
        '''if (leak_candidate['adc_sum_end0'].iloc[0] < row['adc_sum_end0']) and \
           (leak_candidate['adc_sum_end1'].iloc[0] < row['adc_sum_end1']):
            candidates.append(leak_candidate)'''
        candidates.append(leak_candidate)
    return pd.concat(candidates) if candidates else None


def check_signal_bleed_neighbour_once_removed(MIP_df, signal_df):
    signal_df = signal_df.set_index(['pf_event', 'layer', 'strip'])
    leak_candidates_df = pd.DataFrame()
    leak_candidates_df = MIP_df.apply(lambda row: __check_neighboring_bars_once_removed(row, signal_df, leak_candidates_df), axis=1)
    return leak_candidates_df.reset_index()


def __check_neighboring_bars_once_removed(row, signal_df, leak_candidates_df):
    candidates = []
    for neighbor_strip in [row['strip'] - 2, row['strip'] + 2]:  # Check neighboring strips
        try:
            key = (row['pf_event'], row['layer'], neighbor_strip)
            leak_candidate = signal_df.loc[[key]].reset_index()
        except KeyError:
            continue

        if (leak_candidate['adc_sum_end0'].iloc[0] < row['adc_sum_end0']) and \
                (leak_candidate['adc_sum_end1'].iloc[0] < row['adc_sum_end1']):
            candidates.append(leak_candidate)

    return pd.concat(candidates) if candidates else None


def plot_overlapping_hists(data_df1, single_channel_df, save_to_file_path=None):
    layer = single_channel_df['layer'].iloc[0]
    strip = single_channel_df['strip'].iloc[0]
    data_filter = (data_df1['layer'] == layer) & (data_df1['strip'] == strip)
    filtered_data = data_df1[data_filter]
    for end in range(2):
        fig = plt.figure(num=1, clear=True)
        ax1 = fig.add_subplot()
        filtered_data.hist('adc_sum_end' + str(end), ax=ax1, bins=600, range=[0, 6000], log=True, color='b', label='full data')
        single_channel_df.hist('adc_sum_end' + str(end), ax=ax1, bins=600, range=[0, 6000], log=True, color='r', label='leak candidates')
        ax1.set_title("layer: " + str(layer) + " bar: " + str(strip) + " end: " + str(end))
        ax1.set_xlabel("ADC sum")
        ax1.set_ylabel("nbr channels")
        if save_to_file_path is not None:
            plt.savefig(save_to_file_path + "_layer_" + str(layer) + "_bar_" + str(strip) + "_end_" + str(end) + ".pdf")
        else:
            plt.show()


def corr_hist_to_csv(corr_hist, file_path):
    row_list = []
    col_list = []
    data_list = []
    for row in range(corr_hist.shape[0]):
        for col in range(corr_hist.shape[1]):
            row_list.append(row)
            col_list.append(col)
            data_list.append(corr_hist[row, col])
    d = {'row': row_list, 'col': col_list, 'data': data_list}
    frame = pd.DataFrame(data=d)
    frame.to_csv(file_path)


def df_to_corr_hist(data_df):
    hist = np.zeros([192, 192])
    for row in range(192):
        for col in range(192):
            data_selection = (data_df['row'] == row) & (data_df['col'] == col)
            selected_data = data_df[data_selection]
            hist[row, col] = selected_data['data'].iloc[0]
    return hist


def identify_colinear_MIP_tracks(tagged_MIPs):
    print('identifying MIP tracks...')
    event_max = max(tagged_MIPs['pf_event'])
    selected_mip_tracks = tagged_MIPs.groupby(['pf_event']).apply(lambda group:
                                                            __line_up_MIP_tracks(group, event_max))
    selected_mip_tracks = selected_mip_tracks.reset_index(drop=True)
    return selected_mip_tracks


def __line_up_MIP_tracks(group, event_max):
    if int(group.name) % 100 == 0:
        print(int(group.name)/event_max * 100, '%')

    # convert front half bars into back half indexation
    result = group.copy()
    result.loc[result['layer'] <= 9, 'strip'] += 2

    # define track in odd and even layers
    odd_layer_filter = result['layer'] % 2 == 1
    even_layer_filter = result['layer'] % 2 == 0
    odd_layers = result[odd_layer_filter]
    even_layers = result[even_layer_filter]

    if len(even_layers) == 0:
        return even_layers
    elif len(odd_layers) == 0:
        return odd_layers

    result['MIP_odd_layer_position'] = stats.mode(odd_layers['strip'].to_numpy())[0]
    result['MIP_even_layer_position'] = stats.mode(even_layers['strip'].to_numpy())[0]

    # convert front half bars back into front half indexation
    result.loc[result['layer'] <= 9, 'strip'] -= 2

    return result


def toa_stats(tagged_MIPs, plot_single_event=False, plot_single_bar=False):
    result = None
    if plot_single_event:
        result = tagged_MIPs.groupby(['pf_event']).apply(lambda group:
                                                        __get_TOA_spread_event(group, plot_single_event))
    elif plot_single_bar:
        result = tagged_MIPs.groupby(['layer', 'strip']).apply(lambda group:
                                                         __get_TOA_spread_bar(group, plot_single_bar))
    return result


def __get_TOA_spread_event(group, plot_single_event):
    if plot_single_event:
        fig, ax = plt.subplots(1,1)
        group.hist(column=['toa_end0'], bins=50, ax=ax, label='end 0', color='g', histtype='step')
        group.hist(column=['toa_end1'], bins=50, ax=ax, label='end 1', color='r', histtype='step')
        ax.legend()
        plt.show()


def __get_TOA_spread_bar(group, plot_single_bar):
    if plot_single_bar:
        layer, bar = group.name
        fig, ax = plt.subplots(1,1)
        group.hist(column=['toa_end0'], bins=50, ax=ax, label='end 0', color='g', histtype='step')
        group.hist(column=['toa_end1'], bins=50, ax=ax, label='end 1', color='r', histtype='step')
        ax.set_title('layer ' + str(layer) + ' bar ' + str(bar))
        ax.legend()
        plt.show()


def toa_diff_stats(tagged_MIPs, plot_single_event=False, plot_single_bar=False, track_information=False):
    result = None
    if plot_single_event:
            result = tagged_MIPs.groupby(['pf_event']).apply(lambda group:
                                                         __get_TOA_diff_event(group, plot_single_event))

    elif plot_single_bar:
        if not track_information:
            result = tagged_MIPs.groupby(['layer', 'strip']).apply(lambda group:
                                                               __get_TOA_diff_bar(group, plot_single_bar))
        else:
            result = tagged_MIPs.groupby(['MIP_odd_layer_position', 'MIP_even_layer_position']).apply(lambda group:
                                                                                                      __get_TOA_diff_bar_with_track(group))

    result = result.reset_index(drop=True)

    return result


def __get_TOA_diff_event(group, plot_single_event):
    if plot_single_event:
        fig, ax = plt.subplots(1,1)
        group['diff'] = group['toa_end0'] - group['toa_end1']
        group.hist(column=['diff'], bins=50, ax=ax, label='toa diff', color='g')
        ax.legend()
        plt.show()


def __get_TOA_diff_bar(group, plot_single_bar):
    if plot_single_bar:
        layer, bar = group.name
        fig, ax = plt.subplots(1,1)
        group['diff'] = group['toa_end0'] - group['toa_end1']
        group.hist(column=['diff'], bins=50, ax=ax, label='toa diff', color='g', range=[-30, 30])
        ax.set_title('layer ' + str(layer) + ' bar ' + str(bar))
        ax.legend()
        plt.show()


def __get_TOA_diff_bar_with_track(group):
    odd_layer_bar, even_layer_bar = group.name
    group['diff'] = group['toa_end0'] - group['toa_end1']
    for layer in range(min(group['layer']), max(group['layer'])):
        if layer % 2 == 1:
            if layer <= 9:
                bar = odd_layer_bar - 2
                title = 'TOA diff in layer ' + str(layer) + ' bar ' + str(bar) + ' track: odd layer bars ' + str(bar) + ' even layer bars ' + str(even_layer_bar - 2)
            else:
                bar = odd_layer_bar
                title = 'TOA diff in layer ' + str(layer) + ' bar ' + str(bar) + ' track: odd layer bars ' + str(bar) + ' even layer bars ' + str(even_layer_bar)
            bar_filter = (group['layer'] == layer) & (group['strip'] == bar)
        else:
            if layer <= 9:
                bar = even_layer_bar - 2
                title = 'TOA diff in layer ' + str(layer) + ' bar ' + str(bar) + ' track: odd layer bars ' + str(odd_layer_bar - 2) + ' even layer bars ' + str(bar)
            else:
                bar = even_layer_bar
                title = 'TOA diff in layer ' + str(layer) + ' bar ' + str(bar) + ' track: odd layer bars ' + str(odd_layer_bar) + ' even layer bars ' + str(bar)
            bar_filter = (group['layer'] == layer) & (group['strip'] == bar)

        fig, ax = plt.subplots(1, 1)
        bar_data = group[bar_filter]
        bar_data.hist(column=['diff'], bins=50, ax=ax, label='toa diff', color='g', range=[-25, 25])
        ax.set_title(title)
        plt.show()


def compare_TOA_same_channel(tagged_MIPs):
    result = tagged_MIPs.groupby(['layer', 'strip']).apply(lambda group:
                                                     __get_TOA_comparisons_bar(group))
    return result


def __get_TOA_comparisons_bar(group):
    layer, strip = group.name
    fig, ax = plt.subplots(1, 1)
    cmap = plt.get_cmap('coolwarm')
    if layer % 2 == 1:
        if layer <= 9:
            track_filter = group['MIP_odd_layer_position'] - 2 == strip
        else:
            track_filter = group['MIP_odd_layer_position'] == strip
        data = group[track_filter]
        n = max(data['MIP_even_layer_position']) - min(data['MIP_even_layer_position'])
        color_idx = 0
        for i in range(int(min(data['MIP_even_layer_position'])), int(max(data['MIP_even_layer_position']))):
            color = cmap(color_idx/n)
            color_idx += 1
            bar_data_filter = data['MIP_even_layer_position'] == i
            bar_data = data[bar_data_filter]
            bar_data['diff'] = bar_data['toa_end0'] - bar_data['toa_end1']
            bar_data = realign_toa_diff(bar_data)

            if layer <= 9:
                bar_data.hist(column=['diff'], bins=50, ax=ax, label='track in bar ' + str(i - 2), color=color,
                              range=[-15, 15], histtype='step', linewidth=2, density=True)
            else:
                bar_data.hist(column=['diff'], bins=50, ax=ax, label='track in bar ' + str(i), color=color,
                              range=[-15, 15], histtype='step', linewidth=2, density=True)
            ax.set_title('TOA diffs in bar ' + str(strip) + ' layer ' + str(layer))
        ax.legend()
        plt.show()

    else:
        if layer <= 9:
            track_filter = group['MIP_even_layer_position'] - 2 == strip
        else:
            track_filter = group['MIP_even_layer_position'] == strip
        data = group[track_filter]
        n = max(data['MIP_odd_layer_position']) - min(data['MIP_odd_layer_position'])
        color_idx = 0
        for i in range(int(min(data['MIP_odd_layer_position'])), int(max(data['MIP_odd_layer_position']))):
            color = cmap(color_idx / n)
            color_idx += 1
            bar_data_filter = data['MIP_odd_layer_position'] == i
            bar_data = data[bar_data_filter]
            bar_data['diff'] = bar_data['toa_end0'] - bar_data['toa_end1']
            bar_data = realign_toa_diff(bar_data)

            if layer <= 9:
                bar_data.hist(column=['diff'], bins=50, ax=ax, label='track in bar ' + str(i - 2), color=color,
                              range=[-15, 15], histtype='step', linewidth=2, density=True)
            else:
                bar_data.hist(column=['diff'], bins=50, ax=ax, label='track in bar ' + str(i), color=color,
                              range=[-15, 15], histtype='step', linewidth=2, density=True)
            ax.set_title('TOA diffs in bar ' + str(strip) + ' layer ' + str(layer))
        ax.legend()
        plt.show()


def calibrate_TOA(tagged_MIPs):
    result = tagged_MIPs.groupby(['layer', 'strip']).apply(lambda group:
                                                           __get_TOA_bar_calibration(group))
    return result


def __get_TOA_bar_calibration(group):
    layer, strip = group.name
    print('Calibrating layer ' + str(layer) + ' bar ' + str(strip))
    result = pd.DataFrame()
    fits = pd.DataFrame()
    if layer % 2 == 1:
        if layer <= 9:
            track_filter = group['MIP_odd_layer_position'] - 2 == strip
        else:
            track_filter = group['MIP_odd_layer_position'] == strip
        data = group[track_filter]

        for i in range(int(min(data['MIP_even_layer_position'])), int(max(data['MIP_even_layer_position']))):

            bar_data_filter = data['MIP_even_layer_position'] == i
            bar_data = data[bar_data_filter]
            bar_data['diff'] = bar_data['toa_end0'] - bar_data['toa_end1']
            bar_data = realign_toa_diff(bar_data)

            data_vector = bar_data['diff'].to_numpy().flatten()
            mean, std_dev = stats.norm.fit(data_vector)
            fits = pd.concat([fits, pd.DataFrame({'intersection': [i], 'mean': [mean], 'std_dev': [std_dev]})])

    else:
        if layer <= 9:
            track_filter = group['MIP_even_layer_position'] - 2 == strip
        else:
            track_filter = group['MIP_even_layer_position'] == strip
        data = group[track_filter]

        for i in range(int(min(data['MIP_odd_layer_position'])), int(max(data['MIP_odd_layer_position']))):

            bar_data_filter = data['MIP_odd_layer_position'] == i
            bar_data = data[bar_data_filter]
            bar_data['diff'] = bar_data['toa_end0'] - bar_data['toa_end1']
            bar_data = realign_toa_diff(bar_data)

            data_vector = bar_data['diff'].to_numpy().flatten()
            mean, std_dev = stats.norm.fit(data_vector)
            fits = pd.concat([fits, pd.DataFrame({'intersection': [i], 'mean': [mean], 'std_dev': [std_dev]})])

    fits = fits.dropna()
    if fits.empty:
        print('Skipped die to insufficient data')
        return
    fit_intersection = fits['intersection'].to_numpy().flatten()
    fit_x = __intersection_to_meters(fit_intersection)
    fit_y = fits['mean'].to_numpy().flatten()
    fit_s = fits['std_dev'].to_numpy().flatten()
    if (len(fit_y) < 2):
        print('Skipped die to insufficient data')
        return
    line_fit, pcov = curve_fit(__linear_f, fit_x, fit_y, sigma=fit_s)
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(fit_x, fits['mean'].to_numpy().flatten(), yerr=fits['std_dev'].to_numpy().flatten(), linestyle='none', marker='o', color='black', linewidth=1)
    x = np.linspace(0, 2, 20)
    label = 'fit: y = ' + str(line_fit[0]) + 'x + ' + str(line_fit[1])
    y = __linear_f(x, line_fit[0], line_fit[1])
    ax.plot(x, y, label=label, linewidth=2, linestyle='dotted', color='r')
    ax.set_title('TOA calibration layer ' + str(layer) + ' bar ' + str(strip))
    ax.set_ylabel('TOA diff (end 0 - end 1) [ns]')
    ax.set_xlabel('Location along the bar [m]')
    ax.legend()
    plt.savefig('C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/calibrations/plots/TOA_calibrations/TOA_calib_layer' + str(layer) + 'bar' + str(strip) + '.png')
    plt.close()
    result = pd.concat([result, pd.DataFrame({'lin_fit_k': [line_fit[0]], 'lin_fit_m': [line_fit[1]]})])
    return result


def __intersection_to_meters(intersections):
    strip_width = 0.05
    nbr_strips = 2 / 0.05
    intersection_map = {}
    intersections += 14
    for i in range(int(nbr_strips)):
            intersection_map[str(i)] = i * strip_width + strip_width / 2
    result = []
    for i in intersections:
        result.append(intersection_map[str(i)])
    return result


def __linear_f(x, k, m):
    return k * x + m



def calculate_adc_max_pedestals(pedestal_df):
    pedestal_df['adc_max_threshold'] = pedestal_df.apply(__make_adc_max_pedestal, axis=1)
    return pedestal_df


def __make_adc_max_pedestal(row):
    p = 0.999999875
    return stats.norm.ppf(p, row['pedestal_per_time_sample_mean'], row['pedestal_per_time_sample_std_dev'])


def realign_toa_diff(data_df):
    data_df.loc[data_df['diff'] < -25, 'diff'] += 25
    data_df.loc[data_df['diff'] > 25, 'diff'] -= 25
    return data_df


if __name__ == '__main__':
    quit()

