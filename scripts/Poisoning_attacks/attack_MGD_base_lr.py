# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.datasimulation import DataSimulation
from utils.argparsers.simulationargparser import SimulationArgumentParser
from algorithms.PDGD.pdgd import PDGD
from algorithms.PDGD.deeppdgd import DeepPDGD
from algorithms.DBGD.tddbgd import TD_DBGD
from algorithms.DBGD.pdbgd import P_DBGD
from algorithms.DBGD.tdmgd import TD_MGD
from algorithms.DBGD.pmgd import P_MGD
from algorithms.baselines.pairwise import Pairwise
from algorithms.DBGD.neural.pdbgd import Neural_P_DBGD
from algorithms.DBGD.pdbgd_dsp import P_DBGD_DSP
from algorithms.DBGD.pmgd_dsp import P_MGD_DSP
from algorithms.DBGD.tdNSGD import TD_NSGD
from algorithms.DBGD.tdNSGD_dsp import TD_NSGD_DSP

description = 'Run script for testing framework.'
parser = SimulationArgumentParser(description=description)

rankers = []

# Baselines
ranker_params = {
  'learning_rate_decay': 0.9999977}

sim_args, other_args = parser.parse_all_args(ranker_params)
run_name = 'attack/TD_MGD' 

rankers.append((run_name, TD_MGD, other_args))

sim = DataSimulation(sim_args)
sim.run(rankers)
