import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep


# README! For this plotting script to work you need to assign the paths for
# the data file in which the histogram is stored (the files in the "MIP_efficiency_curves" directory)
# and the path to the directory where you wish to save the final figure found below!

hist_data_file_path = "PATH to data file"
save_path = "PATH to save directory"


def makeLabel():
    hep.cms.text(exp="Experiment", text="Internal", fontsize=11, loc=0)


if __name__ == '__main__':
    # Plot style setup
    plt.style.use(hep.style.ROOT)

    # Make figures 3.5 inches wide
    figureWidth = 3.5

    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['xaxis.labellocation'] = 'center'
    mpl.rcParams['yaxis.labellocation'] = 'center'

    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['xtick.minor.size'] = 3
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.minor.size'] = 3
    mpl.rcParams['ytick.major.size'] = 5

    mpl.rcParams['legend.fontsize'] = 8

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    # Let's choose a consistent color scheme for data and MC
    dataColor = colors[0]
    mcColor = 'black'
    unfilColor = 'orange'
    oddColor = colors[1]
    oddColor2 = colors[2]

    print(colors)


    # Read in data
    hist_data = pd.read_csv(hist_data_file_path)


    # Prepare plotting objects
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(figureWidth, 2.5)


    # Format data
    data = hist_data['data'].to_numpy().flatten()
    MC_data = hist_data['MC'].to_numpy().flatten()
    bins = hist_data['bins'].to_numpy().flatten()

    # Plot data
    n, bins1, patches = plt.hist(bins, len(bins), weights=data, histtype='step', label="Data", color=dataColor)
    n2, bins2, patches2 = plt.hist(bins, len(bins), weights=MC_data, histtype='step', label="MC", color=mcColor, linestyle="--")
    plt.vlines(5, 0, 1.5, label=r"Pedestal + 5$\sigma$", color=oddColor2)

    plt.xlim(0, 150)
    plt.ylim(0, 1.2)
    plt.xlabel(r"Readout Threshold [$\sigma$ Above Pedestal]")
    plt.ylabel("MIP Detection Efficiency")
    plt.title('')
    makeLabel()


    plt.legend(loc=0)
    plt.savefig(
        save_path,
        bbox_inches="tight")
    plt.show()