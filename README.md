# HCAL Testbeam Analysis Framework
This is a Python-centric framework that will allow users to implement a full chain of calibrations and analysis on decoded ROOT files collected during the CERN testbeam campaign in 2022.

It is assumed that a large portion of this framework will ultimately be implemented directly into LDMX-sw and possibly transcribed into the C++ chain. However, this package is meant to accelerate analysis and to serve as a baseline.

## Directory Structure
* analysis_files/
  - This is where the output data files from the chain are kept
  - Empty by default
* calibrations/
  - Calibrations produced by and used by the framework are placed here
  - Populated with previously calculated TOA, pedestal, and MIP calibrations by default
* data/
  - Decoded ROOT files can be placed in this directory (not required, but full path should be given to input data when required)
  - See [this repo to generate decoded ROOT files](https://github.com/LDMX-Software/ldmx-tb-online)
  - Empty by default
* notebooks/
  - Example Jupyter Notebooks that demonstrate how to interact with this framework
* plots/
  - Directory for holding plots generated within the framework
  - Empty by default
* src/
  - Source files that work-horse this framework (all written in Python)
 
## Workflow
1. Reformat decoded ROOT files
    Transform input ROOT files into csv files for ease of use. This involves removing unneeded information and aggregating individual time-sample data into per-event information. The goal is to reduce the file size and reformat.
    - Source file: makeAnalysisFiles.py
    - Notebook: reformat_root_files.ipynb
  
2. Calculate pedestal
    Calculate the pedestals by selecting events where the TOA is exactly zero on both ends of a bar. This assumes you are using run 287 (defocused muons) and will fail if you are not. The output is a csv file containing the pedestal information for use in analysis. **This only needs to be done once, and a pre-calculated pedestal file is included in this package.**
    - Source file: calculatePedestals.py
    - Notebook: pedestal_calculation.ipynb
  
3. Calculate MIPs
    Calculate the MIPs by selecting events where the TOA is exactly non-zero on both ends of a bar. This assumes you are using run 287 (defocused muons) and will fail if you are not. The output is a csv file containing the MIP information for use in analysis. **This only needs to be done once, and a pre-calculated MIP file is included in this package.**
    - Source file: calculateMIPs.py
    - Notebook: mip_calculation.ipynb
  
4. Perform data selection and do analysis
    Select events where the TOA is non-zero on both ends of the bar and there is exactly one MIP event in the first layer (to remove pre-showering events). In addition, the sum of ADC is recalculated by subtracting the pedestal and transforming into MIP equivalents. The included Jupyter Notebook provides examples on how to utilize the output for analysis purposes.
    - Source file: selectCleanEvents.py
    - Notebook: clean_and_analyze.ipynb
  
## To-do
1. Implement code that performs alignment on the two halves of the HCAL prototype. **As of now, the code assumes this has not been done!**
2. Include code that calibrates TOA (local on Joe's machine).
3. For pulse-fitting, include functionality to retain time-sample ADC information
4. Add functionality for calibrated TOT (when studied and complete).
  
##### Contact Joe at jmuse@umn.edu (or at joe.m.muse@gmail.com) with any questions or concerns.