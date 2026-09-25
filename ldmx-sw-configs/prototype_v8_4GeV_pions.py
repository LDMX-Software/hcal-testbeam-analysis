import argparse,sys
from LDMX.Framework import ldmxcfg

"""
Simulation of particles through TB prototype
"""

parser = argparse.ArgumentParser(f'ldmx fire {sys.argv[0]}')
parser.add_argument('--nevents',default=5000,type=int)
parser.add_argument('--particle',default='pi-') # other options, mu-,e-,pi-,proton
parser.add_argument('--energy',default=4.0,type=float)
parser.add_argument('--raw_pedestals',default=None,type=str)
arg = parser.parse_args()

p = ldmxcfg.Process('sim')
p.maxEvents = arg.nevents
p.termLogLevel = 0
p.logFrequency = 1
p.verbosity = 0
#NTuples are saved to this "histogram" file
p.histogramFile = "pionHistFilev8_"+str(arg.energy)+"GeV.root"
p.seed = 2

detector = 'ldmx-hcal-prototype-v2.0' # TODO: CHANGE TO FEFIX version

p.outputFiles = [
        arg.particle
        +"Simv8_%.2fGeV_"%arg.energy
        + str(p.maxEvents)
        + "_%s.root"%detector
        ]

from LDMX.SimCore import simulator
import LDMX.Ecal.EcalGeometry # geometry required by sim

mySim = simulator.simulator('mySim')
mySim.setDetector(detector)

# Get a pre-written generator
from LDMX.SimCore import generators as gen

myGPS = gen.gps( 'myGPS' , [
            "/gps/particle " + str(arg.particle),
            "/gps/pos/type Plane",
            "/gps/pos/shape Circle",
            "/gps/direction 0 0 1",
            "/gps/pos/centre 0 0 -600 mm",
            "/gps/pos/radius 25 mm", #Beamspot size
            "/gps/ene/mono " + str(arg.energy) + " GeV"
            ] )
mySim.generators = [ myGPS ]

p.sequence.append( mySim )
mySim.verbosity = 0
mySim.verbose = 0

# import chip/geometry (hardcoded) conditions
import LDMX.Hcal.HcalGeometry
import LDMX.Hcal.digi as hcal_digi
from LDMX.DQM import dqm
from LDMX.Hcal.DetectorMap import HcalDetectorMap
import LDMX.Hcal.hgcrocFormat as hcal_format
import LDMX.Hcal.digi as hcal_digi
import LDMX.Hcal.HcalGeometry
import LDMX.Hcal.hcal_hardcoded_conditions
from LDMX.DQM import dqm
from LDMX.Packing import rawio

import os
#Load channel map
detmap = HcalDetectorMap(f'{os.environ["LDMX_BASE"]}/ldmx-sw/Hcal/data/testbeam_connections.csv')

digi = hcal_digi.HcalDigiProducer()
#No zero suppression in TB data
digi.zeroSuppression = False
#Necessary to produce ADC-TOT cross-calibrations
digi.savePulseTruthInfo = True

import LDMX.Hcal.hcal_testbeamsim_conditions
from LDMX.Conditions.SimpleCSVTableProvider import SimpleCSVIntegerTableProvider, SimpleCSVDoubleTableProvider

# for generating a channel-indexed conditions table
from libDetDescr import HcalID, HcalDigiID
import pandas as pd

# find potential existing conditions file
#working_directory = os.path.dirname(os.path.abspath(__file__))
#clean_conditions_file = working_directory + '/DetID_idx_conditions_v6.csv'

from LDMX.Tools import HgcrocEmulator

#HGCROCEmulator pulse shape loosely matched to data
digi.hgcroc.rateUpSlope = -0.08
digi.hgcroc.rateDnSlope = 0.013
digi.hgcroc.nADCs = 8
digi.hgcroc.iSOI = 1

HcalHgcrocConditionsHardcode=SimpleCSVDoubleTableProvider("HcalHgcrocConditions", [
                "PEDESTAL",
                "NOISE",
                "MEAS_TIME",
                "PAD_CAPACITANCE",
                "TOT_MAX",
                "DRAIN_RATE",
                "GAIN",
                "READOUT_THRESHOLD",
                "TOA_THRESHOLD",
                "TOT_THRESHOLD"
            ])

#HGCROCEmulator conditions
#   - To match ADC and TOT gain
#   - Pedestal position
#   - TOT saturation
HcalHgcrocConditionsHardcode.validForAllRows([
        100. , #PEDESTAL
        1.15869978984993, #NOISE - 0.02 PE with 1 PE ~ 5mV and gain = 1.2
        #-37.5, #MEAS_TIME - ns - clock_cycle/2 - defines the point in the BX where an in-time (time=0 in times vector) hit would arrive
        25, #MEAS_TIME - ns - clock_cycle/2 - defines the point in the BX where an in-time (time=0 in times vector) hit would arrive 
        2.55, #PAD_CAPACITANCE - pF
        200., #TOT_MAX - ns - maximum time chip would be in TOT mode
        90., #DRAIN_RATE - fC/ns - dummy value for now //first at 63
        2.2, #GAIN - large ADC gain for now - conversion from ADC to mV
        5, #READOUT_THRESHOLD - 4 ADC counts above pedestal
        6.2, #TOA_THRESHOLD - mV - 1 PE above pedestal ( 1 PE  - 5 mV conversion)
        1160., #TOT_THRESHOLD - mV - very large for now
        ])

# add them to the sequence
p.sequence.extend(
    [
        digi,
        dqm.NtuplizeHgcrocDigiCollection(
            input_name = 'HcalDigis',
            pedestal_table = None,
            using_eid=False,
            already_aligned=False,
            save_truth=True,
            input_truth_name= 'HcalPulseTruth'
        ),
        dqm.HgcrocPulseTruth(
            input_digi_name="HcalDigis",
            input_truth_name="HcalPulseTruth"
        ),
    ]
)

