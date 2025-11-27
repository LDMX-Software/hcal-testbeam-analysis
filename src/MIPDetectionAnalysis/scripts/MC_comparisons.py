import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':

    data_file = 'C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/spike_fraction.csv'
    MC_file = 'C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/trigger_fraction.csv'

    data_df = pd.read_csv(data_file)
    MC_df = pd.read_csv(MC_file)

    fig1, ax1 = plt.subplots(1, 1)
    fig2, ax2 = plt.subplots(1, 1)
    fig3, ax3 = plt.subplots(1, 1)

    data_df.hist(column=['fraction'], bins=100, range=[0, 1], ax=ax1)
    MC_df.hist(column=['fraction'], bins=100, range=[0, 1], ax=ax2)
    data_df.hist(column=['fraction'], bins=60, range=[0, 0.6], ax=ax3, label='drop errors', density=True)
    MC_df.hist(column=['fraction'], bins=60, range=[0, 0.6], ax=ax3, label='trigger errors', histtype='step', linewidth=2, color='black', density=True)
    ax3.set_yscale('log')
    ax3.set_title('Fraction of channels experiencing readout errors')
    ax3.set_xlabel(r'Fraction of channels')
    ax3.set_ylabel('Logarithmic density')
    #ax3.ticklabel_format(useOffset=False, style='plain')

    ax3.legend()


    plt.show()
