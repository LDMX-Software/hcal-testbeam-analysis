"""Decoding configuration for raw testbeam data
@author Tom Eichlersmith, University of Minnesota
@author Erik Wallin, Lund University
"""

import argparse, sys

parser = argparse.ArgumentParser(f'ldmx fire {sys.argv[0]}',
    description=__doc__)

parser.add_argument('--input_file_0', type=str)
parser.add_argument('--input_file_1', type=str)
parser.add_argument('--pause',action='store_true')
parser.add_argument('--max_events',type=int)
parser.add_argument('--pedestals',default=None,type=str)

arg = parser.parse_args()

from LDMX.Framework import ldmxcfg

p = ldmxcfg.Process('decode')
if arg.max_events is not None :
    p.maxEvents = arg.max_events
p.termLogLevel = 0
p.logFrequency = 1000

import LDMX.Hcal.hgcrocFormat as hcal_format
import LDMX.Hcal.HcalGeometry
import LDMX.Hcal.hcal_hardcoded_conditions
from LDMX.DQM import dqm
import os
from LDMX.Hcal.DetectorMap import HcalDetectorMap
detmap = HcalDetectorMap(f'{os.environ["LDMX_BASE"]}/ldmx-sw/Hcal/data/testbeam_connections.csv')
detmap.want_d2e = True # helps quicken the det -> elec translation                                                                                                                                                                                                                                                                                                   

# extract and deduce parameters from input file name                                                                                                                                                                                                                                                                                                                 

params = os.path.basename(arg.input_file_0).replace('.raw','').split('_')
run = params[params.index("run")+1]
day = params[-2]
time = params[-1]

dir_name  = f'{os.environ["LDMX_BASE"]}/testbeam/aligned/v2-decoded'
os.makedirs(dir_name, exist_ok=True)
os.makedirs(dir_name+'-ntuple', exist_ok=True)

file_name = f'decoded_aligned_run_{run}_{day}_{time}'

p.outputFiles = [f'{dir_name}/{file_name}.root']
p.histogramFile = f'{dir_name}-ntuple/ntuple_{file_name}.root'


# sequence                                                                                                                                                                                                                                                                                                                                                           
#   1. decode event packet into digi collection                                                                                                                                                                                                                                                                                                                      
#   2. ntuplize digi collection                                                                                                                                                                                                                                                                                                                                      
p.sequence = [
        hcal_format.HcalRawDecoder(
            input_file = arg.input_file_0,
            output_name = "HCalDigisUnalignedFPGA0"
            ),
        hcal_format.HcalRawDecoder(
            input_file = arg.input_file_1,
            output_name = "HCalDigisUnalignedFPGA1"
            ),
        hcal_format.HcalAlignPolarfires(
            "HCalDigisAligned",
            ["HCalDigisUnalignedFPGA0","HCalDigisUnalignedFPGA1"],
            drop_lonely_events=False
            ),
        dqm.NtuplizeHgcrocDigiCollection(
            input_name = "HCalDigisAligned",
            pedestal_table = arg.pedestals,
            already_aligned=True,
            is_simulation=True
            )
        ]

if arg.pause :
    p.pause()

