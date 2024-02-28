import pandas as pd
import numpy as np
import uproot
import os
import scipy.stats as stats
from statistics import mean,stdev,median
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
import mplhep as hep

hep.style.use(hep.style.ATLAS)

# Main class to calculate and save pedestals
class calculatePedestals:
    def __init__(self, root_file_name, out_directory='../calibrations/', plot_pedestals=True, plots_directory='../plots/pedestals', do_one_bar=False):
        try:
            self.run_number = root_file_name.split('/')[-1].split('_')[5]
        except:
            self.run_number = root_file_name.split('_')[5]
        if(self.run_number != '287'):
            raise ValueError('Expecting run 287 for 4 GeV defocused muons!!')
        self.root_file_name = root_file_name + ':ntuplizehgcroc/hgcroc'
        self.out_directory = out_directory
        self.do_one_bar = do_one_bar
        self.plot_pedestals = plot_pedestals
        with uproot.open(self.root_file_name) as in_file:
            self.in_data = in_file.arrays(["layer","end","strip","adc","tot","toa","pf_event"], library="pd")
        self.out_ped_individ = {}
        self.out_ped_sum = {'layer': [],
                'strip': [],
                'end': [],
                'pedestal': [],
                'mean': [],
                'std_dev': [],
                'pedestal_per_time_sample': []}

        # Create the directory if it doesn't exist
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)

        # Check plots directory and create if needed
        if plot_pedestals is True:
            if plots_directory is None:
                self.plots_directory = self.out_directory
            else:
                self.plots_directory = plots_directory
                if not os.path.exists(self.plots_directory):
                    # Create the directory if it doesn't exist
                    os.makedirs(self.plots_directory)

    # Properly index events in DataFrame manipulation step
    def __pivot_dataframe(self, dataframe, end):

        # Do some trickery to properly index the events
        multiindex_df = dataframe[dataframe['end'] == end].set_index(['pf_event', dataframe[dataframe['end'] == end].groupby('pf_event').cumcount()])
        
        # Pivot the DataFrame to make 'adc' values into separate columns for each end
        pivoted_df = multiindex_df['adc'].unstack().add_prefix('adc_end_'+str(end)).reset_index()

        return pivoted_df

    # Properly format DataFrame in DataFrame manipulation step
    def __format_dataframe(self, dataframe, end):

        # Perform aggregation, where we extract one value for TOT, TOA and all 8 values of ADC per event
        aggregated_end = dataframe[dataframe['end'] == end].groupby('pf_event').agg({
            'layer': 'first',
            'strip': 'first',
            'tot': 'sum',
            'toa': 'sum',
            'adc': 'sum'
        }).reset_index()

        # Do some manipulation to merge both ends of the bar
        aggregated_end.columns = ['pf_event', 'layer', 'strip', 'tot', 'toa','adc_sum']

        return aggregated_end

    # Clean DataFrame for easy handling
    def __clean_dataframes(self, aggregated_end0, aggregated_end1):
        df_ = pd.merge(aggregated_end0, aggregated_end1, on='pf_event', suffixes=('_end0', '_end1'))
        
        df_ = df_.drop('layer_end0', axis=1)
        df_ = df_.drop('strip_end0', axis=1)

        df_.rename(columns={'layer_end1': 'layer'}, inplace=True)
        df_.rename(columns={'strip_end1': 'strip'}, inplace=True)

        return df_

    # Calculate per-time sample pedestals (to be subtracted off each individual time sample ADC)
    def __get_individual_pedestals(self, group):
        layer, bar = group.name

        index0 = group.columns.str.contains(r'adc_end_0')
        end_0 = group.iloc[:, index0].to_numpy().flatten()

        index1 = group.columns.str.contains(r'adc_end_1')
        end_1 = group.iloc[:, index1].to_numpy().flatten()

        self.out_ped_individ['layer_'+str(layer)+'_bar_'+str(bar)+'_end_0'] = stats.mode(end_0)[0]
        self.out_ped_individ['layer_'+str(layer)+'_bar_'+str(bar)+'_end_1'] = stats.mode(end_1)[0]

    def __plot_pedestal(self, dataframe, layer, bar, end, mean, std_dev):
        # Define figure for end and plot
        fig,ax = plt.subplots(figsize=(8, 8))
        
        ax.hist(dataframe,bins=100,density=True,histtype='step',range=[-200,400])
            
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mean, std_dev)
        
        ax.plot(x, p, 'k', linewidth=2)
        
        ax.set_yscale('log')
        ax.set_xlabel('Sum ADC '+str(end))
        ax.set_ylabel('Events')
        ax.set_ylim(1e-8,1e-1)
        ax.legend()
        
        ax.set_title('Layer: '+str(layer)+' Bar: '+str(bar)+' Side: 0 '+str(round(mean,2))+' '+str(round(std_dev,2)))
        
        plt.savefig(self.plots_directory+'/pedestal_ped_subtracted_side_'+str(end)+'_layer_'+str(layer)+'_bar_'+str(bar)+'.pdf')
        plt.close()

    # Calculate sum of ADC pedestals (to be subtracted off the case of all 8 time samples added)
    def __get_sum_pedestals(self, group):
        layer, bar = group.name

        # Obtain pedestal appropriate for summation of all 8 time samples
        pedestal_temp0 = 8 * self.out_ped_individ['layer_'+str(layer)+'_bar_'+str(bar)+'_end_0']
        pedestal_temp1 = 8 * self.out_ped_individ['layer_'+str(layer)+'_bar_'+str(bar)+'_end_1']

        # Apply pedestal to sum of ADC events
        group['adc_sum_end0'] = group['adc_sum_end0']-(pedestal_temp0)
        group['adc_sum_end1'] = group['adc_sum_end1']-(pedestal_temp1)

        # Select pedestal region for plotting
        fit0_ = (group['adc_sum_end0']<200)&(group['adc_sum_end0']>-200)
        fit1_ = (group['adc_sum_end1']<200)&(group['adc_sum_end1']>-200)
        fit0 = group[fit0_]
        fit1 = group[fit1_]
        
        # Fit a Gaussian to the pedestals
        mean0, std_dev0 = stats.norm.fit(fit0['adc_sum_end0'])
        mean1, std_dev1 = stats.norm.fit(fit1['adc_sum_end1'])

        self.out_ped_sum['layer'].append(layer)
        self.out_ped_sum['strip'].append(bar)
        self.out_ped_sum['end'].append(0)
        self.out_ped_sum['pedestal'].append(pedestal_temp0)
        self.out_ped_sum['mean'].append(mean0)
        self.out_ped_sum['std_dev'].append(std_dev0)
        self.out_ped_sum['pedestal_per_time_sample'].append(self.out_ped_individ['layer_'+str(layer)+'_bar_'+str(bar)+'_end_0'])

        self.out_ped_sum['layer'].append(layer)
        self.out_ped_sum['strip'].append(bar)
        self.out_ped_sum['end'].append(1)
        self.out_ped_sum['pedestal'].append(pedestal_temp1)
        self.out_ped_sum['mean'].append(mean1)
        self.out_ped_sum['std_dev'].append(std_dev1)
        self.out_ped_sum['pedestal_per_time_sample'].append(self.out_ped_individ['layer_'+str(layer)+'_bar_'+str(bar)+'_end_1'])

        # Make plots
        if self.plot_pedestals is True:
            self.__plot_pedestal(group['adc_sum_end0'], layer, bar, 0, mean0, std_dev0)
            self.__plot_pedestal(group['adc_sum_end1'], layer, bar, 1, mean1, std_dev1)

    def __process_group(self, group):
        layer, bar = group.name

        print('layer: ', layer, ', bar: ', bar)

        # Manipulation
        pivoted_df_0 = self.__pivot_dataframe(group, 0)
        pivoted_df_1 = self.__pivot_dataframe(group, 1)

        pivoted_df = pd.merge(pivoted_df_0, pivoted_df_1, on='pf_event').reset_index()

        aggregated_end0 = self.__format_dataframe(group, 0)
        aggregated_end1 = self.__format_dataframe(group, 1)

        result_df = self.__clean_dataframes(aggregated_end0, aggregated_end1)

        result_df = pd.merge(result_df,pivoted_df, on='pf_event').reset_index()

        return result_df

    # Main function to calculate pedestals. Creates a csv that has both pedestals appropriate for individual time samples as well as for the sum of ADC
    def get_pedestals(self):

        # If we only want to look at one bar, define this here
        if self.do_one_bar is True:
            self.in_data = self.in_data[(self.in_data['layer']==1) & (self.in_data['strip']==3)]

        # First, we have to create a temporary DataFrame that keeps all 8 time samples to calculate a pedestal to subtract off individual time samples
        # Process each layer and bar independently
        grouped_data = self.in_data.groupby(['layer', 'strip'], group_keys=False)

        final_result_df = grouped_data.apply(self.__process_group)

        # Select likely pedestal events where the TOA and TOT are both zero on each end of the bar
        selection = (final_result_df['tot_end0']==0) & (final_result_df['tot_end1']==0) & (final_result_df['toa_end0']==0) & (final_result_df['toa_end1']==0)

        final_result_df = final_result_df[selection]

        # Get individual pedestals
        grouped_data_individual = final_result_df.groupby(['layer', 'strip'], group_keys=False)

        grouped_data_individual.apply(self.__get_individual_pedestals)

        # Now we will get the sum of ADC-appropriate pedestals
        grouped_data_sum = final_result_df.groupby(['layer', 'strip'], group_keys=False)

        grouped_data_sum.apply(self.__get_sum_pedestals)

        # Create our final DataFrame with all pedestals
        ped_df = pd.DataFrame(self.out_ped_sum)

        # Save our pedestals to a csv file
        ped_df.to_csv(self.out_directory+'/pedestals.csv', index=False)