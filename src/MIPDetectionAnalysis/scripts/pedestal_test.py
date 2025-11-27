import sys
sys.path.append('../src')

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_file_name = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/adc_pedestal_DPM1_20220424_220542.root"
    pedestals = calculatePedestals(data_file_name, plot_pedestals=False, plots_directory='C:/Users'
                    '/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/pedestals/plots',
                                   out_directory='C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/Calibrations', layer_wise=True,
                                   in_batches=True, )
    pedestals.get_pedestals_no_beam()


