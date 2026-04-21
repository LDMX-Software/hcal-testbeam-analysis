import math
import shutil

from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import os


if __name__ == '__main__':
    MIP_effs_file = "C:/Users/axelh/Desktop/LDMX/LDMX_Data_analysis_project/hcal_testbeam_analysis_main/analysis_files/run_analysis_fpga_project/True_MIPs_filtered/True_MIP_depth_3_filtered_efficiencies_golden_channels.csv"

    MIP_effs = pd.read_csv(MIP_effs_file)
    bp1_filter = MIP_effs['layer'] <= 11
    bp2_filter = MIP_effs['layer'] > 11
    tagged_MIPs_bp1 = MIP_effs[bp1_filter]
    tagged_MIPs_bp2 = MIP_effs[bp2_filter]

    fig, ax = plt.subplots(1, 1)

    tagged_MIPs_bp1.hist(column=['MIP_efficiency'], bins=101, range=[0, 1.01], ax=ax, label='Back plane 1', color='g')
    tagged_MIPs_bp2.hist(column=['MIP_efficiency'], bins=101, range=[0, 1.01], ax=ax, label='Back plane 2', color='r')

    HGCROC1_filter = MIP_effs['layer'] <= 4
    HGCROC2_filter = (MIP_effs['layer'] > 4) & (MIP_effs['layer'] <= 8)
    HGCROC3_filter = (MIP_effs['layer'] > 8) & (MIP_effs['layer'] <= 11)
    HGCROC4_filter = (MIP_effs['layer'] > 11) & (MIP_effs['layer'] <= 14)
    HGCROC5_filter = (MIP_effs['layer'] > 14) & (MIP_effs['layer'] <= 16)
    HGCROC6_filter = MIP_effs['layer'] > 16

    tagged_MIPs_HGCROC1 = MIP_effs[HGCROC1_filter]
    tagged_MIPs_HGCROC2 = MIP_effs[HGCROC2_filter]
    tagged_MIPs_HGCROC3 = MIP_effs[HGCROC3_filter]
    tagged_MIPs_HGCROC4 = MIP_effs[HGCROC4_filter]
    tagged_MIPs_HGCROC5 = MIP_effs[HGCROC5_filter]
    tagged_MIPs_HGCROC6 = MIP_effs[HGCROC6_filter]

    fig2, ax2 = plt.subplots(1, 1)

    tagged_MIPs_HGCROC1.hist(column=['MIP_efficiency'], bins=301, range=[0.6, 1.01], ax=ax2, label='HGCROC 1', color='g')
    tagged_MIPs_HGCROC2.hist(column=['MIP_efficiency'], bins=301, range=[0.6, 1.01], ax=ax2, label='HGCROC 2', color='r')
    tagged_MIPs_HGCROC3.hist(column=['MIP_efficiency'], bins=301, range=[0.6, 1.01], ax=ax2, label='HGCROC 3', color='b')
    tagged_MIPs_HGCROC4.hist(column=['MIP_efficiency'], bins=301, range=[0.6, 1.01], ax=ax2, label='HGCROC 4', color='y')
    tagged_MIPs_HGCROC5.hist(column=['MIP_efficiency'], bins=301, range=[0.6, 1.01], ax=ax2, label='HGCROC 5', color='purple')
    tagged_MIPs_HGCROC6.hist(column=['MIP_efficiency'], bins=301, range=[0.6, 1.01], ax=ax2, label='HGCROC 6', color='orange')


    ax.legend()
    ax2.legend()
    plt.show()



