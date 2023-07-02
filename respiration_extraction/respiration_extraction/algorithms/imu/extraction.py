from abc import abstractmethod

import pandas as pd
import numpy as np
import math
from tpcp import Algorithm, make_action_safe
from scipy.signal import savgol_filter


class IMUBaseExtraction(Algorithm):
    """Base Class defining Interface for other Algorithm Classes
    Takes:  IMU-Data
    Result: Saves resulting respiration Signal as a instance attribute
    """

    _action_methods = "extract"

    # __init__ should be subclass specific
    def __int__(self):
        # Result
        self.respiratory_signal: pd.DataFrame = None

    # Interface Method
    @make_action_safe
    @abstractmethod
    def extract(self, imu_signal: pd.DataFrame, sampling_rate: float):
        pass


class SavGolExtractionGyr(IMUBaseExtraction):

    """
    Extraction algorithm using a SavGol Filter on gyration
    based on https://doi.org/10.1109/EMBC44109.2020.9176245
    """

    def __init__(self):
        self.respiratory_signal = None

    def extract(self, imu_signal: pd.DataFrame, sampling_rate: float):

        # Calculate Spectra for calculating window size
        summed_power_spectrum = 0
        for col in imu_signal:
            power_spectrum = np.abs(np.fft.rfft(imu_signal[col])) ** 2
            summed_power_spectrum = summed_power_spectrum + power_spectrum

        #  take all Spectra in the Range of 0 - 2Hz for calculating window size
        freqs = np.fft.rfftfreq(len(imu_signal), 1 / sampling_rate)
        freqs.sort()
        power = pd.DataFrame({"Power": summed_power_spectrum}, index=freqs)
        sliced_power_spectrum = power[0.001:2.0]

        # find global maximum which represents respiration rate which is used for calculating windowsize
        respiration_frequency = sliced_power_spectrum["Power"].idxmax()

        # Determine Windwow size
        window_size_seconds = 1 / respiration_frequency
        window_size_samples = math.floor(window_size_seconds * sampling_rate)

        # Filter with Savitsky-Golay filter
        gyr_signal = pd.DataFrame(
            savgol_filter(
                # paper suggest acceleration downwards (acc_x in paper and acc y for NilsPod)  and gyroscope
                # (acc_ in paper and acc y for NilsPod) to the right side of the subject
                # As Portabiles
                imu_signal["gyr_x"],
                polyorder=4,
                window_length=window_size_samples,
            ),
            index=imu_signal.index,
        )

        # filter with moving average filter with window size of 0.75s
        moving_window_size = math.floor(0.75 * sampling_rate)
        gyr_signal = gyr_signal.rolling(moving_window_size, min_periods=1).mean()

        # normalize
        normalized_gyr = (gyr_signal - gyr_signal.min()) / (
            gyr_signal.max() - gyr_signal.min()
        )
        # save
        self.respiratory_signal = normalized_gyr
        return self


class SavGolExtractionAcc(IMUBaseExtraction):

    """
    Extraction algorithm using a SavGol Filter on acceleration
    based on https://doi.org/10.1109/EMBC44109.2020.9176245
    """

    def __init__(self):
        self.respiratory_signal = None

    def extract(self, imu_signal: pd.DataFrame, sampling_rate: float):

        # Calculate Spectra for calculating window size
        summed_power_spectrum = 0
        for col in imu_signal:
            power_spectrum = np.abs(np.fft.rfft(imu_signal[col])) ** 2
            summed_power_spectrum = summed_power_spectrum + power_spectrum

        #  take all Spectra in the Range of 0 - 2Hz for calculating window size
        freqs = np.fft.rfftfreq(len(imu_signal), 1 / sampling_rate)
        freqs.sort()
        power = pd.DataFrame({"Power": summed_power_spectrum}, index=freqs)
        sliced_power_spectrum = power[0.001:2.0]

        # find global maximum which represents respiration rate which is used for calculating windowsize
        respiration_frequency = sliced_power_spectrum["Power"].idxmax()

        # Determine Windwow size
        window_size_seconds = 1 / respiration_frequency
        window_size_samples = math.floor(window_size_seconds * sampling_rate)

        # Filter with Savitsky-Golay filter
        acc_signal = pd.DataFrame(
            savgol_filter(
                # paper suggest acceleration downwards (acc_x in paper and acc y for NilsPod)  and gyroscope to the right side of the subject
                # As Portabiles
                imu_signal["acc_y"],
                polyorder=4,
                window_length=window_size_samples,
            ),
            index=imu_signal.index,
        )

        # filter with moving average filter with window size of 0.75s
        moving_window_size = math.floor(0.75 * sampling_rate)
        acc_signal = acc_signal.rolling(moving_window_size, min_periods=1).mean()

        # normalize
        normalized_acc = (acc_signal - acc_signal.min()) / (
            acc_signal.max() - acc_signal.min()
        )

        # save
        self.respiratory_signal = normalized_acc
        return self


class PositionalVectorExtraction(Algorithm):

    """
    Algorithm for extracting a respiration-like Positional Vektor Signal from the acceleration respiratory_signal in x,y
    and z directions
    Takes:  acceleration in x,y,z directions as a pd.Dataframe with each direction as column
    Result: Saves resulting respiration Signal in self.respiratory_signal and the respiration rate in self.respiration_rate
    """

    def __int__(self):
        self.respiratory_signal = None
        self.respiration_spline = None
        self.respiration_rate = None

    def extract(self, imu_signal: pd.DataFrame, sampling_rate: float):
        # Paper used window size = 10 and  sampling rate = 10 Hz thus we use a window size equal to the sampling rate
        window_size = math.floor(sampling_rate)
        imu_filtered = imu_signal.rolling(window_size, min_periods=1).mean()

        # Min-Max normalize the signals
        normalized_imu = (imu_filtered - imu_filtered.min()) / (
            imu_filtered.max() - imu_filtered.min()
        )

        # Use another rolling average with size 15 in the paper thus 1.5 * sampling size here
        window_size = math.floor(sampling_rate * 1.5)
        discrete_signal = normalized_imu.rolling(window_size, min_periods=1).mean()

        # Calculate Amplitude
        discrete_signal["Amplitude"] = discrete_signal.sum(
            axis=1, numeric_only=True
        ) ** (1 / 3)

        # normalize the final signal
        normalized_discrete_signal = pd.DataFrame(
            discrete_signal["Amplitude"] - discrete_signal["Amplitude"].min()
        ) / (discrete_signal["Amplitude"].max() - discrete_signal["Amplitude"].min())

        # save and return
        self.respiratory_signal = normalized_discrete_signal
        return self
