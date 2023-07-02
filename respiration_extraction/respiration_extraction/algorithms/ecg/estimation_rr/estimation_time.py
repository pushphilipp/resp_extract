import pandas as pd
import neurokit2 as nk
import numpy as np
import scipy.signal
from scipy.signal import argrelextrema

from respiration_extraction.algorithms.ecg.estimation_rr.base_estimation import (
    BaseEstimationRR,
)


# E1
class PeakDetection(BaseEstimationRR):

    """Implement basic breath detection algorithm from
    Shah S A 2012 Vital sign monitoring and data fusion for paediatric triage PhD Thesis University of Oxford
    https://ora.ox.ac.uk/objects/uuid:80ae66e3-849b-4df1-b064-f9eb7530200d
    """

    def __init__(self):
        self.respiration_rate = None

    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):
        """
        @param respiration_signal: respiratory_signal to extract respiration rate
        @param sampling_rate: sampling rate of the respiratory_signal
        @return: self, respiration rate is saved as a instance attribute self.respiration_rate (float if successful nan
         otherwise)
        """
        # Find local maxima of filtered respiratory_signal
        # Number of Points in original Paper was only 1 or 2 if flat peak -> adjusted to 3 because higher sampling rate
        number_points = 3

        resp_peaks = argrelextrema(
            np.array(respiration_signal.iloc[:, 0]),
            np.greater_equal,
            order=number_points,
        )[0]

        duration = len(respiration_signal.index) / sampling_rate / 60.0
        peak_count = len(resp_peaks)

        if duration != 0:
            self.respiration_rate = peak_count / duration
        else:
            self.respiration_rate = np.nan
        return self


# E2
class GradiantDetection(BaseEstimationRR):

    """Implements Gradiant detection algorithm from https://doi.org/10.1007/BF02348427"""

    def __init__(self):
        self.respiration_rate = None

    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):
        """
        @param respiration_signal: respiratory_signal to extract respiration rate (must be 0-1 normalized)!!!
        @param sampling_rate: sampling rate of the respiratory_signal
        @return: self, respiration rate is saved as a instance attribute self.respiration_rate (float if successful nan
        otherwise)
        """
        # normalize to -0.5 to 0.5
        respiration_signal = respiration_signal - 0.5
        zero_crossings = np.where(
            np.diff(np.sign(np.array(respiration_signal.iloc[:, 0])))
        )[0]

        # Duration between detected breaths (Charlton did this in his matlab code)
        samples = zero_crossings[-1] - zero_crossings[0]
        duration = samples / sampling_rate / 60.0

        # Detected breaths
        # Divided by 2 because only positive zero crossing should be detected (each positive is followed by a negative)
        breath_count = len(zero_crossings) / 2

        # respiration rate
        if duration != 0:
            self.respiration_rate = breath_count / duration
        else:
            self.respiration_rate = np.nan
        return self


# E3
class PeakThroughDetection(BaseEstimationRR):
    """
    Implements simple breath detection algorithm from:
    Fleming, S. 2010. “Measurement and Fusion of Non-Invasive Vital Signs for Routine Triage of Acute Paediatric Ill
    ness.” PhD thesis, Oxford University, UK.
    """

    def __init__(self):
        self.respiration_rate = None

    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):
        """
        @param respiration_signal: respiratory_signal to extract respiration rate
        @param sampling_rate: sampling rate of the respiratory_signal
        @return: self, respiration rate is saved as a instance attribute self.respiration_rate (float if successful nan otherwise)
        """
        # Filter the respiratory_signal of interest by a band-pass filter with passband of 0.1–0.5 Hz
        filtered_respiration = pd.DataFrame(
            {
                "filtered": nk.signal_filter(
                    signal=respiration_signal.iloc[:, 0],
                    sampling_rate=sampling_rate,
                    lowcut=0.1,
                    highcut=0.5,
                    order=10,
                )
            }
        )
        # Find local maxima and minima of filtered respiratory_signal
        peaks, _ = scipy.signal.find_peaks(filtered_respiration["filtered"])
        throughs, _ = scipy.signal.find_peaks(
            np.negative(filtered_respiration["filtered"])
        )

        # Defining Threshold for peaks and throughs
        threshold_mean = filtered_respiration.iloc[0, :].mean(
            skipna=True,
            axis=0,
        )

        # find indices of minima
        minima_erased = (
            filtered_respiration.loc[
                (filtered_respiration.filtered >= threshold_mean)
                & ([index in throughs for index in filtered_respiration.index])
            ]
            .dropna()
            .index
        )
        # delete minimas
        throughs = throughs[[through not in minima_erased for through in throughs]]
        # remove the closest peak to the removed minima
        for minimum in minima_erased:
            distances = minimum - peaks
            closest_peak_index = np.absolute(distances).argmin()
            peaks = np.delete(peaks, closest_peak_index)

        # find indices of maxima
        maxima_erased = (
            filtered_respiration.loc[
                (filtered_respiration.filtered <= threshold_mean)
                & ([index in peaks for index in filtered_respiration.index])
            ]
            .dropna()
            .index
        )

        # delete maxima
        peaks = peaks[[peak not in maxima_erased for peak in peaks]]

        # remove the closest peak to the removed minima
        for maxima in maxima_erased:
            distances = maxima - throughs
            closest_through_index = np.absolute(distances).argmin()
            throughs = np.delete(throughs, closest_through_index)

        # Eliminate peaks which are within 0.5s after the previous peak (KEEP The first peak!)
        # Watchout! Antipattern ahead!
        # Also this is very rare to happen (not yet happened during testing) so it could also be left out
        time_window = 0.5  # In seconds!
        for maximum in peaks:
            next_index = maximum + (time_window * sampling_rate)
            window = filtered_respiration.loc[maximum:next_index, :]
            # If there is a peak remove all peaks and throughs in this timewindow
            if len(window.loc[[index in peaks for index in window.index], :]) > 1:
                # remove peaks
                peaks = np.delete(
                    peaks,
                    [
                        peak
                        for peak in peaks
                        if ((peak > maximum) & (peak <= next_index))
                    ],
                )
                # remove throughs
                throughs = np.delete(
                    throughs,
                    [
                        through
                        for through in throughs
                        if ((through > maximum) & (through <= next_index))
                    ],
                )

                print(
                    "Deleted Peaks and throughs in {} second Timewindow in the peak-through estimation algorithm. This should be very rare if this happens more frequently, maybe check your signal quality.  ".format(
                        time_window
                    )
                )

        # Calculate RR from peaks
        duration = len(filtered_respiration.index) / sampling_rate / 60.0
        peak_count = len(peaks)
        if duration != 0:
            self.respiration_rate = peak_count / duration
        else:
            self.respiration_rate = np.nan
        return self


# E4
class CountOrig(BaseEstimationRR):
    """
    Original Counting Method from Schäfer, A., Kratky, K.W. Estimation of Breathing Rate from Respiratory Sinus Arrhythmia:
    Comparison of Various Methods. Ann Biomed Eng 36, 476–485 (2008). https://doi.org/10.1007/s10439-007-9428-1
    """

    def __init__(self):
        self.respiration_rate = None

    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):
        """
        @param respiration_signal: respiratory_signal to extract respiration rate
        @param sampling_rate: sampling rate of the respiratory_signal
        @return: self, respiration rate is saved as a instance attribute self.respiration_rate (float if successful nan
        otherwise)
        """
        # Filter the respiratory_signal of interest by a band-pass filter with passband of 0.1–0.5 Hz
        filtered_respiration = pd.DataFrame(
            {
                "filtered": nk.signal_filter(
                    signal=respiration_signal.iloc[:, 0],
                    sampling_rate=sampling_rate,
                    lowcut=0.1,
                    highcut=0.5,
                    order=10,
                )
            }
        )

        filtered_respiration = filtered_respiration - filtered_respiration.mean()
        # Find local maxima and minima of filtered respiratory_signal
        peaks, _ = scipy.signal.find_peaks(filtered_respiration["filtered"])
        throughs, _ = scipy.signal.find_peaks(
            np.negative(filtered_respiration["filtered"])
        )

        filtered_respiration["maxima"] = filtered_respiration.iloc[peaks]["filtered"]
        filtered_respiration["minima"] = filtered_respiration.iloc[throughs]["filtered"]
        # Take third quartile (Q3) of all local maximum ordinate values
        q3 = filtered_respiration["maxima"].quantile(0.75)
        # Define 0.2 * Q3 as threshold
        threshhold = 0.2 * q3
        # Find all cycles
        # cycle := begin and end at local maxima above threshold level and contains
        # exactly one minimum below zero and no other local extrema

        # Filter maxima
        maxima = (
            filtered_respiration[filtered_respiration["maxima"] > threshhold]
            .drop("minima", axis=1)
            .dropna()
            .reset_index()
        )
        # filter minima
        minima = (
            filtered_respiration[filtered_respiration["minima"] < 0]
            .drop("maxima", axis=1)
            .dropna()
        )

        # List with a tuple for each cycle peak-peak ( number of troughs, duration )
        cycles = [
            (
                len(minima[(peak < minima.index) & (next_peak > minima.index)]),
                next_peak - peak,
            )
            for peak, next_peak in zip(
                maxima["index"], maxima["index"].shift(-1).dropna()
            )
        ]
        # Take only cycles with one trough
        relevant_cycles = [cycle for cycle in cycles if cycle[0] == 1]

        # Estimate RR: The average length of all detected respiratory cycles
        # is interpreted as the reciprocal respiration frequency.
        if len(relevant_cycles) != 0:
            average_sampling_points = sum(
                [cycle[1] for cycle in relevant_cycles]
            ) / len(relevant_cycles)
            average_duration = average_sampling_points / sampling_rate
            self.respiration_rate = 60 / average_duration
            return self
        else:
            self.respiration_rate = np.nan
            return self


# E5
class CountAdvDetection(BaseEstimationRR):
    """
    Advanced Counting Method from Schäfer, A., Kratky, K.W. Estimation of Breathing Rate from Respiratory Sinus Arrhythmia:
    Comparison of Various Methods. Ann Biomed Eng 36, 476–485 (2008). https://doi.org/10.1007/s10439-007-9428-1
    """

    def __init__(self):
        self.respiration_rate = None

    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):
        """
        @param respiration_signal: respiratory_signal to extract respiration rate
        @param sampling_rate: sampling rate of the respiratory_signal
        @return: self, respiration rate is saved as a instance attribute self.respiration_rate (float if successful nan
        otherwise)
        """
        # Filter the respiratory_signal of interest by a band-pass filter with passband of 0.1–0.5 Hz
        filtered_respiration = pd.DataFrame(
            {
                "filtered": nk.signal_filter(
                    signal=respiration_signal.iloc[:, 0],
                    sampling_rate=sampling_rate,
                    lowcut=0.1,
                    highcut=0.5,
                    order=10,
                )
            }
        )
        # Find local maxima and minima of filtered respiratory_signal
        peaks, _ = scipy.signal.find_peaks(filtered_respiration["filtered"])
        troughs, _ = scipy.signal.find_peaks(
            np.negative(filtered_respiration["filtered"])
        )

        filtered_respiration["maxima"] = filtered_respiration.iloc[peaks]["filtered"]
        filtered_respiration["minima"] = filtered_respiration.iloc[troughs]["filtered"]

        # Define difference threshold
        filtered_respiration["extrema"] = filtered_respiration["minima"].fillna(
            filtered_respiration["maxima"]
        )
        # Calculate all differences between local extrema
        # amplitude from point a to point b is in the same coloumn as point a -> last extrema has NaN
        filtered_respiration["difference"] = abs(
            filtered_respiration["extrema"].dropna()
            - filtered_respiration["extrema"].dropna().shift(-1)
        )
        threshold = 0.3 * filtered_respiration.difference.quantile(0.75)

        # ignore all minima/maxima pairs which have a difference smaller than threshold
        while filtered_respiration.difference.min() < threshold:
            # TODO This could be probably improved a lot by using pandas functionalities / magic
            # Condition for being the follower of the minimum
            m2 = (
                filtered_respiration.difference.dropna().shift(1)
                == filtered_respiration.difference.min()
            )

            # Finding index of follower of minimum
            follower_index = filtered_respiration.difference.dropna().loc[m2].index

            # Remove extrema lable of the follower of minimum (assigning nan to extrema colums)
            filtered_respiration.loc[
                follower_index,
                [
                    "maxima",
                    "minima",
                    "extrema",
                    "difference",
                ],
            ] = np.nan

            # Remove extrema lable of the minimum itself (assigning nan to extrema colums)
            filtered_respiration.loc[
                (
                    filtered_respiration.difference
                    == filtered_respiration.difference.min()
                ),
                [
                    "maxima",
                    "minima",
                    "extrema",
                    "difference",
                ],
            ] = np.nan

        # Irrelevant extrema are now not labeled anymore
        # list of maxima
        maxima = filtered_respiration["maxima"].dropna().reset_index()
        # list of minima
        minima = filtered_respiration["minima"].dropna()

        cycles = [
            (
                len(
                    minima[(peak < minima.index) & (next_peak > minima.index)]
                ),  # Number of troughs between 2 peaks
                next_peak - peak,  # Duration of the cycle
            )
            for peak, next_peak in zip(
                maxima["index"], maxima["index"].shift(-1).dropna()
            )
        ]

        # Estimate RR: The average length of all detected respiratory cycles is interpreted as the reciprocal
        # respiration frequency.
        if len(cycles) != 0:
            average_sampling_points = sum([cycle[1] for cycle in cycles]) / len(cycles)
            average_duration = average_sampling_points / sampling_rate
            self.respiration_rate = 60 / average_duration
            return self
        else:
            self.respiration_rate = np.nan
            return self


class NeurokitDetection(BaseEstimationRR):

    """
    Algorithm used for calculating the reference respiration rate.
    Calculates RR signal and returns the mean of that signal.
    """

    def estimate(self, respiration_signal: pd.DataFrame, sampling_rate: float):

        """
        @param respiration_signal: respiratory_signal to extract respiration rate
        @param sampling_rate: sampling rate of the respiratory_signal
        @return: self, respiration rate is saved as a instance attribute self.respiration_rate (float if successful nan
        otherwise)
        """

        # check if signal is too short
        minimum_seconds = 12
        threshold = sampling_rate * minimum_seconds
        if len(respiration_signal.index) < threshold:
            self.respiration_rate = np.nan
            return self

        # Cleaning signal
        cleaned = nk.rsp_clean(
            respiration_signal.iloc[:, 0], sampling_rate=sampling_rate
        )

        # Extract peaks
        _, peaks = nk.rsp_peaks(cleaned, sampling_rate=sampling_rate)
        rsp_rate = nk.rsp_rate(cleaned, sampling_rate=sampling_rate)

        self.respiration_rate = np.mean(rsp_rate)
        return self
