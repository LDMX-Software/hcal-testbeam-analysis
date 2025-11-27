import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_group(group):
    end, strip, layer = group.name
    print("creating Simple_MIPs: layer: ", layer, "bar: ", strip)
    ax = group.plot.scatter(x='pf_ticks', y='adc')
    ax.set_xlabel('pf_ticks')
    ax.set_ylabel('adc')
    ax.set_title( 'adc as function of pf_ticks in layer ' + str(layer) + ' bar ' + str(strip) + ' end ' + str(end))
    plt.savefig(out_dir + '/time_trend_layer_' + str(layer) + '_bar_' + str(strip) + '_end_' + str(end) + '.pdf')
    plt.show()


if __name__ == '__main__':

    # define files and variables
    data_file_name = "/hcal_testbeam_analysis_main/analysis_files/run_20220425_fpga_287.root"
    out_dir = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/plots/time_trends"
    data_file_name += ':ntuplizehgcroc/hgcroc'
    layers = 19

    # do analysis layer by layer
    for i in range(layers):
        in_data = pd.DataFrame()
        with uproot.open(data_file_name) as in_file:
            print('reading file for layer ', i + 1, '...')
            cut = "layer == " + str(i + 1)
            batchnbr = 1
            for batch in in_file.iterate(
                        ["layer", "end", "strip", "raw_id", "adc", "tot", "toa", "pf_event", "pf_spill",
                        "pf_ticks"], cut, library="pd", step_size="10 MB"):
                print("batch: ", batchnbr)
                in_data = pd.concat([in_data, batch])
                batchnbr += 1


        selection = (in_data['tot'] == 0) & (in_data['toa'] == 0)
        in_data = in_data[selection]
        in_data.to_csv(out_dir + "/test.csv")
        grouped = in_data.groupby(['end', 'strip', 'layer'])
        grouped.apply(plot_group)
