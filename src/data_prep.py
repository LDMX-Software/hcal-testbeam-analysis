# Filename: data_prep.py
# Author: Gréta Gajdán 
# Created: 2025-01-03

import numpy as np
import pandas as pd

def import_data(calibration_folder, data_folder, run_number, is_it_pulsed=False):
    # imports all files needed needed (pedestal, mip and run) 
    
    pedestal_df=pd.read_csv(calibration_folder+"pedestal_fixed.csv", sep=',') # these have the data for both ends in one line
    mip_df=pd.read_csv(calibration_folder+"mip_fixed.csv", sep=',')
    
    if is_it_pulsed==True:
        run_df=pd.read_csv(data_folder+"run_"+str(run_number)+"_pulse.csv", sep=',')
        run_df.drop(["pf_spill", "pf_ticks"], axis=1, inplace=True) # we never need these
        run_df.drop(["toa_end0", "toa_end1"], axis=1, inplace=True) # unreliable guys, but we might need them in the future
    
    else:
        run_df=pd.read_csv(data_folder+"run_"+str(run_number)+".csv", sep=',')
        run_df.drop(["pf_spill", "pf_ticks"], axis=1, inplace=True)
        run_df.drop(["toa_end0", "toa_end1"], axis=1, inplace=True)
        
    return pedestal_df,mip_df,run_df

def drop_extra_pedestal_data(df):
    # drops pedestal daata from merged dataframe. Only works on dataframes that have pedestal and run data merged.
    df.drop(["pedestal_end0", "pedestal_end1",
             "pedestal_per_time_sample_end0","pedestal_per_time_sample_end1"], axis=1, inplace=True)
    return df

def subtract_pedestals(df, is_it_pulsed=False):
    # subtracts pedestal from all columns that have it present. Only works on dataframes that have pedestal and run data merged.
    df["adc_sum_end0"]-=df["pedestal_end0"]
    df["adc_sum_end1"]-=df["pedestal_end1"]
    
    df["adc_max_end0"]-=df["pedestal_per_time_sample_end0"]
    df["adc_max_end1"]-=df["pedestal_per_time_sample_end1"]
    
    df["adc_mean_end0"]-=df["pedestal_per_time_sample_end0"]
    df["adc_mean_end1"]-=df["pedestal_per_time_sample_end1"]

    if is_it_pulsed==True:
        for i in range(8):  # believe it or not, the loop is faster than a wildcard lookup
            adc_end0_col = f'adc_{i}_end0'
            adc_end1_col = f'adc_{i}_end1'
        
            df[adc_end0_col]-=df["pedestal_per_time_sample_end0"]
            df[adc_end1_col]-=df["pedestal_per_time_sample_end1"]
    
    return df

def convert_to_MeV(run_df,mip_df, is_it_pulsed):
    # converts all ADCs to MeV
    merged_df=run_df.merge(mip_df, how='left', on=['layer','strip'])

    end0=merged_df.columns.str.contains("adc_.*_end0")
    end1=merged_df.columns.str.contains("adc_.*_end1")
    
    merged_df[merged_df.columns[end0]]=merged_df[merged_df.columns[end0]].mul(4.66/merged_df["mpv_end0"], axis=0)
    merged_df[merged_df.columns[end1]]=merged_df[merged_df.columns[end1]].mul(4.66/merged_df["mpv_end1"], axis=0)

    merged_df.drop(["mpv_end0", "mpv_end1"], axis=1, inplace=True)
    merged_df.drop(["pedestal_per_time_sample_std_dev_end0", "pedestal_per_time_sample_std_dev_end1"], axis=1, inplace=True)
    
    return merged_df

def select_bars_with_data(run_df, pedestal_df, subtract_pedestal=False, is_it_pulsed=False):
    print("Number of events: "+str(len(run_df.pf_event.unique())))
    print("\n")
    merged_df=run_df.merge(pedestal_df, how='left', on=['layer', 'strip'])

    merged_df=merged_df[(merged_df["adc_sum_end0"]>(merged_df["pedestal_end0"]+5*merged_df["std_dev_end0"])) & 
                        (merged_df["adc_sum_end1"]>(merged_df["pedestal_end1"]+5*merged_df["std_dev_end1"]))]
    
    merged_df.drop(["std_dev_end0", "std_dev_end1"], axis=1, inplace=True)

    if subtract_pedestal==True:
        merged_df=subtract_pedestals(merged_df, is_it_pulsed)

    merged_df=drop_extra_pedestal_data(merged_df) # needs to be dropped regardless of subtraction, too much memory used otherwise
    
    events_left=merged_df.pf_event.unique()
    print("Initial pedestal-based selection performed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return merged_df