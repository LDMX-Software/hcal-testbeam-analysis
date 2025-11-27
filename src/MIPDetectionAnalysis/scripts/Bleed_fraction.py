import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':

    data_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/Calibrations/pedestals_MIP.csv"
    bleed_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/Simple_MIPs/deviant_channels_MIP"
    data_df = pd.read_csv(data_file)
    bleed_df = pd.read_csv(bleed_file)