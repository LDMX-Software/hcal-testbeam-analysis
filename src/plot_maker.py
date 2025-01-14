# Filename: plot_maker.py
# Author: Gréta Gajdán 
# Created: 2025-01-07

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append("../src")

from data_prep import *

def longitudinal_profile(run_df, run_number, plot_folder, save_plot=False):
    # generate the longitudinal shower profile of a run. 
    # only works on dataframes where energy conversion had already happened
    
    n_event=len(run_df.pf_event.unique())
    layer_sum=run_df.groupby(["layer"]).sum()[["adc_sum_end0", "adc_sum_end1"]]/n_event

    plt.bar(layer_sum.index-0.2, layer_sum["adc_sum_end0"], color="magenta", alpha=0.5, edgecolor="magenta", label="end_0", width=0.4)
    plt.bar(layer_sum.index+0.2, layer_sum["adc_sum_end1"], color="blue", alpha=0.5, edgecolor="blue", label="end_1", width=0.4)
    plt.xlim(0.5,19.5)
    plt.xticks(np.arange(1,20))
    plt.legend()
    plt.xlabel("Layer", fontsize=12.5)
    plt.ylabel("Deposited energy [MeV]", fontsize=12.5)
    plt.title("Run "+str(run_number), fontsize=15)
    
    if save_plot==True:
        plt.savefig(plot_folder+"longitudinal_profile.png", bbox_inches='tight')
        
    plt.show()

def transverse_profile(run_df,plot_folder, save_plot=False):
    # generate the transverse shower profile of a run. 
    # only works on dataframes where energy conversion had already happened
    
    n_event=len(run_df.pf_event.unique())
    strip_sum=run_df.groupby(["layer", "strip"]).sum()[["adc_sum_end0", "adc_sum_end1"]]/n_event
    
    y_max=strip_sum.max(axis=None)
    
    for l in range(1,20):
        if l in run_df['layer'].values:
            helper_df=strip_sum.xs(l, level=0, axis=0, drop_level=True)
            
            plt.bar(helper_df.index-0.2, helper_df["adc_sum_end0"], color="magenta", alpha=0.5, edgecolor="magenta", label="end_0", width=0.4)
            plt.bar(helper_df.index+0.2, helper_df["adc_sum_end1"], color="blue", alpha=0.5, edgecolor="blue", label="end_1", width=0.4)
            plt.legend()
            plt.xticks(helper_df.index)
            plt.ylim(5e-4,y_max+1)
            plt.yscale('log')
            plt.xlabel("Bar", fontsize=12.5)
            plt.ylabel("Deposited energy [MeV]", fontsize=12.5)
            plt.title("Layer "+str(l), fontsize=15)
            if save_plot==True:
                Path(plot_folder+"transverse_profile").mkdir(parents=True, exist_ok=True)
                plt.savefig(plot_folder+"transverse_profile/transverse_layer_"+str(l)+".png", bbox_inches='tight')
            plt.show()

def make_shower_profiles(run_df, mip_df, run_number, plot_folder):
    # generate and save both transverse and longitudinal shower profiles of a run
    run_df=convert_to_MeV(run_df, mip_df, is_it_pulsed=False)
    longitudinal_profile(run_df,run_number,plot_folder, save_plot=True)
    transverse_profile(run_df,plot_folder, True)
    return