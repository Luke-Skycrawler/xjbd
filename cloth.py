from params import *

from simulator.base import BaseSimulator
from fem_mesh import FEMMesh



class XPBDSimulator(BaseSimulator): 
    def __init__(self, config_file = "config.json"):
        super().__init__(member_type = FEMMesh, config_file = config_file)