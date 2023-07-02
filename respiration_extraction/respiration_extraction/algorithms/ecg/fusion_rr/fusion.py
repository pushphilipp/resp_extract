from abc import abstractmethod

import numpy as np
import pandas as pd
from tpcp import Algorithm, Parameter, make_action_safe
from typing import Optional, Union, List, Sequence


class BaseFusion(Algorithm):

    """
    Base Class defining Interface for other Fusion Algorithms
    Takes: Number of float Respiration Rates [Breaths per Minute]
    Result: Fused Respiration Rate
    """

    _action_methods = "fuse"

    def __int__(self):
        self.respiration_rate_fused = None

    # Interface Method
    @make_action_safe
    @abstractmethod
    def fuse(self, *resp_rates: float):
        pass


class SmartFusion(BaseFusion):
    """Smart Fusion from Karlen et al 2013"""

    def __init__(self):
        self.respiration_rate_fused = None

    def fuse(self, *resp_rates: float):
        # Calculate Standarddeviation of all RRs
        std = np.std(resp_rates)
        # If std <= 4 bpm then estimate RR as the mean else no Output
        if std <= 4.0:
            self.respiration_rate_fused = np.mean(resp_rates)
        else:
            self.respiration_rate_fused = np.nan
        return self
