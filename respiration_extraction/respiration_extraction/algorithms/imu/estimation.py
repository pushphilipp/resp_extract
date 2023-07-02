# from abc import abstractmethod
#
# import pandas as pd
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# from tpcp import Algorithm, Parameter, make_action_safe
# from scipy.respiratory_signal import savgol_filter, argrelextrema
#
"""
Some of the simple breath detections from IMU-Papers were implemented here. But since the RR estimation algorithms from
respiration_extraction/ecg can also be applied to IMU derived signals they were left out.
"""

#
# class IMU_RR_Estimation(Algorithm):
#     """Base Class defining Interface for other IMU based RR-Estimation Classes
#     Takes:  IMU-derived respiration respiratory_signal
#     Result: Saves resulting respiration rate as an instance attribute
#     """
#
#     _action_methods = "estimate"
#
#     def __int__(self):
#         self.respiration_rate = None
#
#     @make_action_safe
#     @abstractmethod
#     def estimate(self, imu_respiration_signal: pd.DataFrame, sampling_rate: float):
#         pass
#
#
# class PeakEstimation(IMU_RR_Estimation):
#     """Uses a Peak Detection on an IMU-derived Signal to determine a respiration rate based on the RR-Estimation from: doi: 10.1109/EIT51626.2021.9491900.
#     Takes:  IMU-derived respiration respiratory_signal
#     Result: Saves resulting respiration rate as an instance attribute
#     """
#
#     def __init__(self):
#         self.respiration_rate = None
#
#     def estimate(self, imu_respiration_signal: pd.DataFrame, sampling_rate: float):
#         """Taking a Dataframe and doing a RR-Estimation based on the respiratory_signal from the first column"""
#         # Peak Detection
#         number_points = 10
#
#         peaks = imu_respiration_signal.iloc[
#             argrelextrema(
#                 imu_respiration_signal.iloc[:, 0],
#                 np.greater_equal,
#                 order=number_points,
#             )[0]
#         ].iloc[:, 0]
#
#         # Count Peaks during the whole Signal
#         peak_number = len(peaks)
#         duration_seconds = len(imu_respiration_signal) / sampling_rate  # in seconds
#         duration_minutes = duration_seconds / 60.0
#
#         self.respiration_rate = peak_number / duration_minutes
#         return self
