from hcal_testbeam_analysis_main.src.calculatePedestals import *
import pandas as pd
import matplotlib.pyplot as plt


def fit(group):
    layer, bar = group.name
    # fit0_ = (group['adc_mean_end0'] < 300)
    # fit1_ = (group['adc_mean_end1'] < 300)
    # fit0 = group[fit0_]
    # fit1 = group[fit1_]
    mean0, std_dev0 = stats.norm.fit(group['adc_mean_end0'])
    mean1, std_dev1 = stats.norm.fit(group['adc_mean_end1'])



def treat_data(in_data):
    # grouped_data = in_data.groupby(['layer', 'strip'], group_keys=False)
    selection = (in_data['tot_end0'] == 0) & (in_data['tot_end1'] == 0) & (
                in_data['toa_end0'] == 0) & (in_data['toa_end1'] == 0)

    in_data = in_data[selection]

    grouped_data = in_data.groupby(['layer', 'strip'], group_keys=False)
    del in_data
    grouped_data.apply(fit)

    # in_data = in_data[(in_data['layer'] == 2) & (in_data['strip'] == 3)]
    # in_data.hist(column=['adc_mean_end0', 'adc_mean_end1'], bins=149, range=[50, 150])



if __name__ == '__main__':
    file = "C:/Users/axelh/Desktop/LDMX/LDMX Data analysis project/hcal-testbeam-analysis-main/analysis_files/run_20220425_fpga_287.root.csv"
    data = pd.read_csv(file)
    treat_data(data)
    plt.show()



