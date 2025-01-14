# Filename: make_cuts.py
# Author: Gréta Gajdán 
# Created: 2025-01-03

import pandas as pd

def first_layer_cut(df):
    helper_df=df[df['layer']==1].groupby(['pf_event']).sum()
    events_to_remove=helper_df[helper_df['layer']>1].index.values.tolist()
    df=df[~df['pf_event'].isin(events_to_remove)]
    
    events_left=df.pf_event.unique()
    print("Events with multiple hits in first layer removed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return df

def back_layer_cut(df, b_num):
    events_to_remove=df[df['layer']>19-b_num].groupby(['pf_event']).count().index.values.tolist()
    df=df[~df['pf_event'].isin(events_to_remove)]
    
    events_left=df.pf_event.unique()
    print("Events with hits in the back " +str(b_num)+" layers removed.")
    print("Events left: "+str(len(events_left)))
    print("\n")
    return df