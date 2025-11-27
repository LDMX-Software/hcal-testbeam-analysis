import pandas as pd
import uproot


if __name__ == '__main__':
    batchnbr = 1
    with uproot.open('C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/muonDefocusedHistFile_2.0GeV.root:ntuplizehgcroc/hgcroc') as in_file:
        for batch in in_file.iterate(["layer", "end", "strip", "raw_id", "adc", "tot", "toa", "pf_event", "pf_spill", "pf_ticks"], library="pd", step_size="50 MB"):
            batch.to_csv('C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/MC_analysis/MC_data_defocused_muons.csv', index=False)
            print("batch: ", batchnbr)
            batchnbr += 1
