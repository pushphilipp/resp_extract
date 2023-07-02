import pandas as pd
from respiration_extraction.algorithms.ecg.extraction.base_extraction import (
    BaseExtraction,
)
import numpy as np
import neurokit2 as nk

"""contains classes for EDR algorithms from Neurokit2 https://doi.org/10.3758/s13428-020-01516-y"""


class ExtractionSarkar2015(BaseExtraction):

    """
    EDR algorithm 'sarkar2015' from Neurokit2 https://doi.org/10.3758/s13428-020-01516-y
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        ecg = np.array(ecg_signal)[:, 0]
        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
        ecg_rate = nk.ecg_rate(
            rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg)
        )
        edr = nk.ecg_rsp(ecg_rate, sampling_rate=sampling_rate, method="sarkar2015")

        self.respiratory_signal = pd.DataFrame(edr, index=ecg_signal.index)
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionSoni2019(BaseExtraction):

    """
    EDR algorithm 'soni2019' from Neurokit2 https://doi.org/10.3758/s13428-020-01516-y
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        ecg = np.array(ecg_signal)[:, 0]
        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
        ecg_rate = nk.ecg_rate(
            rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg)
        )
        edr = nk.ecg_rsp(ecg_rate, sampling_rate=sampling_rate, method="soni2019")
        self.respiratory_signal = pd.DataFrame(edr, index=ecg_signal.index)
        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self


class ExtractionVangent2019(BaseExtraction):

    """
    EDR algorithm 'vangent2019' from Neurokit2 https://doi.org/10.3758/s13428-020-01516-y
    """

    def extract(self, ecg_signal: pd.DataFrame, sampling_rate: float):
        ecg = np.array(ecg_signal)[:, 0]
        rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate)
        ecg_rate = nk.ecg_rate(
            rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg)
        )
        edr = nk.ecg_rsp(ecg_rate, sampling_rate=sampling_rate, method="vangent2019")
        self.respiratory_signal = pd.DataFrame(edr, index=ecg_signal.index)

        self.respiratory_signal = self.normalize(self.respiratory_signal)
        return self
