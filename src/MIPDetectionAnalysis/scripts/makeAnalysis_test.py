import sys
sys.path.append('../src')

from hcal_testbeam_analysis_main.src.calculatePedestals import *
from hcal_testbeam_analysis_main.src.makeAnalysisFiles import *
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data_file_name = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/Neutrons/neutronDefocusedHistFilev3_2.0GeV.root"
    out_directory =  "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/Neutrons/"
    make_file = make_files = makeAnalysisFiles(data_file_name, out_directory=out_directory,  do_one_bar=False, do_alignment=False, output_pulse_shapes=True)
    make_file.create_dataframes()


