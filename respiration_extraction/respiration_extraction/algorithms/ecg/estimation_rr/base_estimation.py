from abc import abstractmethod

import pandas as pd
from tpcp import Algorithm, Parameter, make_action_safe


class BaseEstimationRR(Algorithm):
    """
    Base Class defining Interface for other Estimation Algorithms
    Takes:  Pandas Dataframe containing the respiration Signal
    Result: Saves resulting respiration rate as attribute in respiration_rate
    """

    _action_methods = "estimate"

    # __init__ should be subclass specific
    def __int__(self):
        self.respiration_rate: float = None

    # Interface Method
    @abstractmethod
    @make_action_safe
    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):
        pass
