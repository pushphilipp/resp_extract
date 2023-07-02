from typing import Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tpcp import Pipeline
from tpcp._dataset import DatasetT
from typing_extensions import Self
from respiration_extraction.algorithms.imu import *
from respiration_extraction.algorithms.ecg import *


class BiopacEDR(Pipeline):
    def __init__(
        self,
        extraction: BaseExtraction,
        estimation: BaseEstimationRR,
    ):
        self.extraction = extraction
        self.estimation = estimation
        self.respiration_rate = None
        self.respiration_signal = None

    def score(self, datapoint: DatasetT) -> Union[float, Dict[str, float]]:

        # start the run method
        self.run(datapoint=datapoint)
        # real respiratory_signal and RR value
        ground_truth_signal = self.normalize(datapoint.respiration)
        ground_truth_estimation = NeurokitDetection()
        # call estimation
        ground_truth_estimation.estimate(
            respiration_signal=ground_truth_signal,
            sampling_rate=datapoint.sampling_rate_biopac,
        )
        rr_value = ground_truth_estimation.respiration_rate

        # resample signals (only one necessary but both are done so they are both an ndarray)
        y = scipy.signal.resample(
            ground_truth_signal.iloc[:, 0], len(self.respiration_signal.index)
        )
        x = scipy.signal.resample(
            self.respiration_signal.iloc[:, 0], len(self.respiration_signal.index)
        )

        correlation = scipy.stats.pearsonr(x, y)[0]
        return {"correlation": correlation, "RR": self.respiration_rate, "GT": rr_value}

    def run(self, datapoint: DatasetT) -> Self:

        # run extraction and estimation
        self.extraction.extract(
            datapoint.biopac_ecg,
            datapoint.sampling_rate_biopac,
        )

        self.estimation.estimate(
            self.extraction.respiratory_signal,
            datapoint.sampling_rate_biopac,
        )
        # save results as instance variables and return self
        self.respiration_signal = self.extraction.respiratory_signal
        self.respiration_rate = self.estimation.respiration_rate
        return self

    def normalize(self, respiration_signal: pd.DataFrame):

        normalized_respiration_signal = (
            respiration_signal - respiration_signal.mean()
        ) / respiration_signal.std()

        scaled_respiration_signal = (
            normalized_respiration_signal - normalized_respiration_signal.min()
        ) / (normalized_respiration_signal.max() - normalized_respiration_signal.min())

        return scaled_respiration_signal


class NilsPodEDR(Pipeline):
    def __init__(
        self,
        extraction: BaseExtraction,
        estimation: BaseEstimationRR,
    ):
        self.extraction = extraction
        self.estimation = estimation
        self.respiration_rate = None
        self.respiration_signal = None

    def score(self, datapoint: DatasetT) -> Union[float, Dict[str, float]]:
        # start the run method
        self.run(datapoint=datapoint)
        # real respiratory_signal and RR value
        ground_truth_signal = self.normalize(datapoint.respiration)
        ground_truth_estimation = NeurokitDetection()
        # call estimation
        ground_truth_estimation.estimate(
            respiration_signal=ground_truth_signal,
            sampling_rate=datapoint.sampling_rate_biopac,
        )
        rr_value = ground_truth_estimation.respiration_rate

        # resample signals (only one necessary but both are done so they are both an ndarray)
        y = scipy.signal.resample(
            ground_truth_signal.iloc[:, 0], len(self.respiration_signal.index)
        )
        x = scipy.signal.resample(
            self.respiration_signal.iloc[:, 0], len(self.respiration_signal.index)
        )

        correlation = scipy.stats.pearsonr(x, y)[0]
        return {"correlation": correlation, "RR": self.respiration_rate, "GT": rr_value}

    def run(self, datapoint: DatasetT) -> Self:

        # run extraction and estimation
        self.extraction.extract(
            datapoint.ecg,
            datapoint.sampling_rate_nilspod,
        )

        self.estimation.estimate(
            self.extraction.respiratory_signal,
            datapoint.sampling_rate_nilspod,
        )
        # save results as instance variables and return self
        self.respiration_signal = self.extraction.respiratory_signal
        self.respiration_rate = self.estimation.respiration_rate
        return self

    def normalize(self, respiration_signal: pd.DataFrame):

        normalized_respiration_signal = (
            respiration_signal - respiration_signal.mean()
        ) / respiration_signal.std()

        scaled_respiration_signal = (
            normalized_respiration_signal - normalized_respiration_signal.min()
        ) / (normalized_respiration_signal.max() - normalized_respiration_signal.min())

        return scaled_respiration_signal


class PosVecPipeline(Pipeline):
    def __int__(self, algorithm: PositionalVectorExtraction):
        self.imu_algorithm = algorithm
        self.processed_imu_signal = None
        self.respiration_rate = None

    def run(self, datapoint: DatasetT) -> Self:
        # Clone Algorithms to avoid data leakage
        imu_algo = self.imu_algorithm.clone()

        # Action methods
        imu_algo.extract(
            imu_signal=datapoint.acc, sampling_rate=datapoint.sampling_rate_nilspod()
        )
        imu_algo.estimate()

        self.respiration_rate = self.imu_algorithm.respiration_rate
        self.processed_imu_signal = self.imu_algorithm.respiratory_signal
        return self

    def score(self, datapoint: DatasetT) -> Union[float, Dict[str, float]]:
        self.run(datapoint)
        # Calculate ground truth respiration rate
        cleaned = nk.rsp_clean(datapoint.respiration, datapoint.sampling_rate_biopac)
        df, peaks_dict = nk.rsp_peaks(
            cleaned, sampling_rate=datapoint.sampling_rate_biopac
        )
        breaths = len(peaks_dict)
        duration = len(cleaned) / datapoint.sampling_rate_biopac

        # RR Difference
        diff = self.respiration_rate

        # Correlation
        cc = scipy.stats.pearsonr()

        return {"correlation": cc, "RR-difference": diff}
        # Calculate


class ImuPipline(Pipeline):
    def __init__(self, extraction: IMUBaseExtraction, estimation: BaseEstimationRR):
        self.imu_extraction = extraction
        self.estimation = estimation
        self.respiration_rate = None
        self.processed_imu_signal = None

    def run(self, datapoint: DatasetT) -> Self:
        # IMU -> Respirationlike Signal
        self.imu_extraction.extract(
            imu_signal=datapoint.nilspod, sampling_rate=datapoint.sampling_rate_nilspod
        )
        # Respiration Signal -> estimated RR
        self.estimation.estimate(
            self.imu_extraction.respiratory_signal, datapoint.sampling_rate_nilspod
        )
        # Save Results in Instance
        self.processed_imu_signal = self.imu_extraction.respiratory_signal
        self.respiration_rate = self.estimation.respiration_rate
        return self

    def score(self, datapoint: DatasetT) -> Union[float, Dict[str, float]]:
        # start the run method
        self.run(datapoint=datapoint)
        # real respiratory_signal and RR value
        ground_truth_signal = self.normalize(datapoint.respiration)
        ground_truth_estimation = NeurokitDetection()
        # call estimation
        ground_truth_estimation.estimate(
            respiration_signal=ground_truth_signal,
            sampling_rate=datapoint.sampling_rate_biopac,
        )
        rr_value = ground_truth_estimation.respiration_rate

        # resample signals (only one necessary but both are done so they are both an ndarray)
        y = scipy.signal.resample(
            ground_truth_signal.iloc[:, 0], len(self.processed_imu_signal.index)
        )
        x = scipy.signal.resample(
            self.processed_imu_signal.iloc[:, 0], len(self.processed_imu_signal.index)
        )

        correlation = scipy.stats.pearsonr(x, y)[0]
        return {"correlation": correlation, "RR": self.respiration_rate, "GT": rr_value}

    def normalize(self, respiration_signal: pd.DataFrame):
        normalized_respiration_signal = (
            respiration_signal - respiration_signal.mean()
        ) / respiration_signal.std()

        scaled_respiration_signal = (
            normalized_respiration_signal - normalized_respiration_signal.min()
        ) / (normalized_respiration_signal.max() - normalized_respiration_signal.min())

        return scaled_respiration_signal
