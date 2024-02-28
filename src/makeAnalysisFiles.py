import pandas as pd
import numpy as np
import uproot
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import mplhep as hep

hep.style.use(hep.style.ATLAS)

from apply_calibrations import *

# Main class to make analysis files (output is csv format)
class makeAnalysisFiles:
    def __init__(self, root_file_name, out_directory='../analysis_files/', calibration_file='../calibrations/toa_calibration_phase3.csv', do_one_bar=False):
        try:
            self.run_number = root_file_name.split('/')[-1].split('_')[5]
        except:
            self.run_number = root_file_name.split('_')[5]
        self.root_file_name = root_file_name + ':ntuplizehgcroc/hgcroc'
        self.out_directory = out_directory
        self.do_one_bar = do_one_bar
        with uproot.open(self.root_file_name) as in_file:
            self.in_data = in_file.arrays(["layer","end","strip","raw_id","adc","tot","toa","pf_event","pf_spill","pf_ticks"], library="pd")
        self.toa_cal_file = pd.read_csv(calibration_file)

        if not os.path.exists(self.out_directory):
            # Create the directory if it doesn't exist
            os.makedirs(self.out_directory)

    # Perform aggregation, where we extract one value for TOT, calibrated TOA and sum, max, mean of all 8 ADC values per event
    def __get_each_end(self, data_frame, end, toa_list):
        aggregated_end = data_frame[data_frame['end'] == end].groupby('pf_event').agg({
            'layer': 'first',
            'strip': 'first',
            'pf_spill': 'first',
            'pf_ticks': 'first',
            'tot': tot_calib,
            'toa': lambda x: toa_calib(x,toa_list,end),
            'adc': ['sum', 'mean', 'max']
        }).reset_index()

        aggregated_end.columns = ['pf_event','layer','strip','pf_spill','pf_ticks','tot','toa','adc_sum','adc_mean','adc_max']

        return aggregated_end

    # Clean per-end DataFrames for useable formats
    def __clean_frame(self, aggregated_end0, aggregated_end1):
        df_ = pd.merge(aggregated_end0, aggregated_end1, on='pf_event', suffixes=('_end0', '_end1'))
        
        df_ = df_.drop('layer_end0', axis=1)
        df_ = df_.drop('strip_end0', axis=1)
        df_ = df_.drop('pf_spill_end0', axis=1)
        df_ = df_.drop('pf_ticks_end0', axis=1)

        df_.rename(columns={'layer_end1': 'layer'}, inplace=True)
        df_.rename(columns={'strip_end1': 'strip'}, inplace=True)
        df_.rename(columns={'pf_spill_end1': 'pf_spill'}, inplace=True)
        df_.rename(columns={'pf_ticks_end1': 'pf_ticks'}, inplace=True)

        return df_

    # Process each layer and bar independently
    def __process_group(self, group):
        layer, bar = group.name

        print('layer: ', layer, ', bar: ', bar)
        toa_cal = (self.toa_cal_file['layer'] == layer - 1) & (self.toa_cal_file['bar'] == bar)
        toa_list = self.toa_cal_file[toa_cal].values.tolist()

        try:
            aggregated_end0 = self.__get_each_end(group, 0, toa_list)
            aggregated_end1 = self.__get_each_end(group, 1, toa_list)

            result_df = self.__clean_frame(aggregated_end0, aggregated_end1)
            return result_df
        except:
            print('empty bar')
            return None

    # Main function
    def create_dataframes(self):
        
        # If we only want to look at one bar, define this here
        if self.do_one_bar is True:
            self.in_data = self.in_data[(self.in_data['layer']==1) & (self.in_data['strip']==3)]

        # Process each layer and bar independently
        grouped_data = self.in_data.groupby(['layer', 'strip'], group_keys=False)

        result_df = grouped_data.apply(self.__process_group)

        # Save to csv file
        result_df.to_csv(self.out_directory + '/run_' + self.run_number + '.csv', index=False)

    # Admittedly dirty function that makes many plots
    def make_plots(self, in_csv_file=None, plots_directory=None):

        # Open csv file
        result_df = None
        if in_csv_file is None:
            result_df = pd.read_csv(self.out_directory + '/run_' + self.run_number + '.csv')
        else:
            result_df = pd.read_csv(in_csv_file)

        # If we only want to look at one bar, define this here
        if self.do_one_bar is True:
            result_df = result_df[(result_df['layer']==1) & (result_df['strip']==3)]

        # Get layers and bars
        layers = result_df['layer'].unique()
        bars = result_df['strip'].unique()

        # Check plots directory and create if needed
        if plots_directory is None:
            plots_directory = self.out_directory
        else:
            if not os.path.exists(plots_directory):
                # Create the directory if it doesn't exist
                os.makedirs(plots_directory)

        # Loop through plots you want to make and plot for each layer/bar (I really don't know why I did it this way)
        for k in range(18):
            print(k)

            # Loop through layers
            for i in range(len(layers)):
                if self.do_one_bar is True and i>0:
                    continue
            
                # Declare layer-level figures
                fig1 = None
                ax1 = None
                if(k<6):
                    fig1,ax1 = plt.subplots(figsize=(8, 8))
            
                in_data_temp_ = result_df[result_df['layer']==layers[i]]

                # Loop through bars
                for j in range(len(bars)):
                    if self.do_one_bar is True and j>0:
                        continue

                    # Declare bar-level figures
                    selection = None
                    fig2 = None
                    ax2 = None
                    if(k>5):
                        fig2,ax2 = plt.subplots(figsize=(8, 8))

                    # Select non-zero TOA events
                    if(k<3):
                        selection = (in_data_temp_['strip']==bars[j]) & (in_data_temp_['toa_end0']>0)& (in_data_temp_['toa_end1']>0)
                    elif(k<12 and k>5):
                        selection = (in_data_temp_['strip']==bars[j]) & (in_data_temp_['toa_end0']>0)& (in_data_temp_['toa_end1']>0)

                    # Select exactly-zero TOA events
                    else:
                        selection = (in_data_temp_['strip']==bars[j]) & (in_data_temp_['toa_end0']==0)& (in_data_temp_['toa_end1']==0)

                    # Impose selection
                    in_data_temp = in_data_temp_[selection]

                    # Non-zero TOA- sum ADC vs TOT (end 0)
                    if(k==6):
                        ax2.hist2d(in_data_temp['adc_sum_end0'],in_data_temp['tot_end0'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('ADC 0')
                        ax2.set_ylabel('TOT 0')
                        ax2.set_title('TOA Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j])+' Side: 0')
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_adc_tot_side_0_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Non-zero TOA- sum ADC vs TOA (end 0)
                    if(k==7):
                        ax2.hist2d(in_data_temp['adc_sum_end0'],in_data_temp['toa_end0'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('ADC 0')
                        ax2.set_ylabel('TOA 0')
                        ax2.set_title('TOA Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j])+' Side: 0')
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_adc_toa_side_0_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Non-zero TOA- TOT vs TOA (end 0)
                    if(k==8):
                        ax2.hist2d(in_data_temp['tot_end0'],in_data_temp['toa_end0'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('TOT 0')
                        ax2.set_ylabel('TOA 0')
                        ax2.set_title('TOA Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j])+' Side: 0')
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_tot_toa_side_0_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Non-zero TOA- sum ADC (end 0) vs sum ADC (end 1)
                    if(k==9):
                        ax2.hist2d(in_data_temp['adc_sum_end0'],in_data_temp['adc_sum_end1'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('ADC 0')
                        ax2.set_ylabel('ADC 1')
                        ax2.set_title('TOA Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j]))
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_adc_adc_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Non-zero TOA- TOA (end 0) vs TOA (end 1)
                    if(k==10):
                        ax2.hist2d(in_data_temp['toa_end0'],in_data_temp['toa_end1'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('TOA 0')
                        ax2.set_ylabel('TOA 1')
                        ax2.set_title('TOA Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j]))
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_toa_toa_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Non-zero TOA- TOT (end 0) vs TOT (end 1)
                    if(k==11):
                        ax2.hist2d(in_data_temp['tot_end0'],in_data_temp['tot_end1'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('TOT 0')
                        ax2.set_ylabel('TOT 1')
                        ax2.set_title('TOA Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j]))
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_tot_tot_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Exactly-zero TOA- sum ADC vs TOT (end 0)
                    if(k==12):
                        ax2.hist2d(in_data_temp['adc_sum_end0'],in_data_temp['tot_end0'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('ADC 0')
                        ax2.set_ylabel('TOT 0')
                        ax2.set_title('TOA Not Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j])+' Side: 0')
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_adc_tot_side_0_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Exactly-zero TOA- sum ADC vs TOA (end 0)
                    if(k==13):
                        ax2.hist2d(in_data_temp['adc_sum_end0'],in_data_temp['toa_end0'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('ADC 0')
                        ax2.set_ylabel('TOA 0')
                        ax2.set_title('TOA Not Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j])+' Side: 0')
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_adc_toa_side_0_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Exactly-zero TOA- TOT vs TOA (end 0)
                    if(k==14):
                        ax2.hist2d(in_data_temp['tot_end0'],in_data_temp['toa_end0'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('TOT 0')
                        ax2.set_ylabel('TOA 0')
                        ax2.set_title('TOA Not Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j])+' Side: 0')
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_tot_toa_side_0_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Exactly-zero TOA- sum ADC (end 0) vs sum ADC (end 1)
                    if(k==15):
                        ax2.hist2d(in_data_temp['adc_sum_end0'],in_data_temp['adc_sum_end1'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('ADC 0')
                        ax2.set_ylabel('ADC 1')
                        ax2.set_title('TOA Not Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j]))
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_adc_adc_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Exactly-zero TOA- TOA (end 0) vs TOA (end 1)
                    if(k==16):
                        ax2.hist2d(in_data_temp['toa_end0'],in_data_temp['toa_end1'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('TOA 0')
                        ax2.set_ylabel('TOA 1')
                        ax2.set_title('TOA Not Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j]))
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_toa_toa_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Exactly-zero TOA- TOT (end 0) vs TOT (end 1)
                    if(k==17):
                        ax2.hist2d(in_data_temp['tot_end0'],in_data_temp['tot_end1'],bins=100,cmin=0.01,cmap='jet',norm=mcolors.LogNorm())
                        ax2.set_xlabel('TOT 0')
                        ax2.set_ylabel('TOT 1')
                        ax2.set_title('TOA Not Fired, Layer: '+str(layers[i])+' Bar: '+str(bars[j]))
                        plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_tot_tot_layer_'+str(layers[i])+'_bar_'+str(bars[j])+'.pdf')
                        plt.close()

                    # Declare a histogram for each layer-level plots (one for each bar)- only considering end 0 since ends are correlated
                    if(k==0):
                        ax1.hist(in_data_temp['adc_sum_end0'],bins=100,histtype='step',label='Bar '+str(bars[j]))
                    if(k==1):
                        ax1.hist(in_data_temp['toa_end0'],bins=100,histtype='step',label='Bar '+str(bars[j]))
                    if(k==2):
                        ax1.hist(in_data_temp['tot_end0'],bins=100,histtype='step',label='Bar '+str(bars[j]))
                    if(k==3):
                        ax1.hist(in_data_temp['adc_sum_end0'],bins=100,histtype='step',label='Bar '+str(bars[j]))
                    if(k==4):
                        ax1.hist(in_data_temp['toa_end0'],bins=100,histtype='step',label='Bar '+str(bars[j]))
                    if(k==5):
                        ax1.hist(in_data_temp['tot_end0'],bins=100,histtype='step',label='Bar '+str(bars[j]))

                # Non-zero TOA- sum ADC all bars (end 0)
                if(k==0):
                    ax1.set_yscale('log')
                    ax1.set_xlabel('ADC 0')
                    ax1.set_ylabel('Events')
                    ax1.legend()
                    ax1.set_title('TOA Fired, Layer: '+str(layers[i]))
                    plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_adc_layer_'+str(layers[i])+'_side_0.pdf')
                    plt.close()

                # Non-zero TOA- TOA all bars (end 0)
                if(k==1):
                    ax1.set_yscale('log')
                    ax1.set_xlabel('TOA 0')
                    ax1.set_ylabel('Events')
                    ax1.legend()
                    ax1.set_title('TOA Fired, Layer: '+str(layers[i]))
                    plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_toa_layer_'+str(layers[i])+'_side_0.pdf')
                    plt.close()

                # Non-zero TOA- TOT all bars (end 0)
                if(k==2):
                    ax1.set_yscale('log')
                    ax1.set_xlabel('TOT 0')
                    ax1.set_ylabel('Events')
                    ax1.legend()
                    ax1.set_title('TOA Fired, Layer: '+str(layers[i]))
                    plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_fired_tot_layer_'+str(layers[i])+'_side_0.pdf')
                    plt.close()

                # Exactly-zero TOA- sum ADC all bars (end 0)
                if(k==3):
                    ax1.set_yscale('log')
                    ax1.set_xlabel('ADC 0')
                    ax1.set_ylabel('Events')
                    ax1.legend()
                    ax1.set_title('TOA Not Fired, Layer: '+str(layers[i]))
                    plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_adc_layer_'+str(layers[i])+'_side_0.pdf')
                    plt.close()

                # Exactly-zero TOA- TOA all bars (end 0)
                if(k==4):
                    ax1.set_yscale('log')
                    ax1.set_xlabel('TOA 0')
                    ax1.set_ylabel('Events')
                    ax1.legend()
                    ax1.set_title('TOA Not Fired, Layer: '+str(layers[i]))
                    plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_toa_layer_'+str(layers[i])+'_side_0.pdf')
                    plt.close()

                # Exactly-zero TOA- TOT all bars (end 0)
                if(k==5):
                    ax1.set_yscale('log')
                    ax1.set_xlabel('TOT 0')
                    ax1.set_ylabel('Events')
                    ax1.legend()
                    ax1.set_title('TOA Not Fired, Layer: '+str(layers[i]))
                    plt.savefig(plots_directory+'/run_'+self.run_number+'_toa_not_fired_tot_layer_'+str(layers[i])+'_side_0.pdf')
                    plt.close()