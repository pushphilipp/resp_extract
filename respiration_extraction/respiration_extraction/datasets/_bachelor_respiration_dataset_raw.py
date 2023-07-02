import csv
import itertools
from pathlib import Path
import biopsykit
import pandas as pd
import re
import locale
import datetime

from matplotlib import pyplot as plt
from tpcp import Dataset
from typing import Optional, Union, List, Sequence
from imucal.management import load_calibration_info


class BachelorRespirationDatasetRaw(Dataset):
    SAMPLING_RATE_NILSPOD: float = 256.0
    SAMPLING_RATE_BIOPAC: float = 250.0

    PHASES: list = [
        "Initialization",
        "Baseline 1",
        "Speaking",
        "Baseline 2",
        "Standing and rest",
        "Sitting and rest",
        "4-7-8 breathing",
        "Baseline 3",
        "Metronome Breathing",
        "Baseline 4",
        "Inhalation and hold",
        "Exhalation and hold",
        "Hyperventilation",
        # "Baseline 5", Last phase excluded because of different lengths and sometimes even empty signal
    ]

    def __init__(
        self,
        data_path: Path,
        exclude_missing: bool = True,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        self.exclude_missing = exclude_missing
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        participant_ids = [
            f.name.split("_")[1] for f in sorted(self.data_path.glob("Subject*/"))
        ]
        if self.exclude_missing:
            participant_ids = [
                pid for pid in participant_ids if pid not in ["01", "04", "15", "16"]
            ]
        else:
            print("Incomplete Subjects included, watch out for missing datastreams")

        index = list(itertools.product(participant_ids, self.PHASES))
        # Create all csv files
        for p_id in participant_ids:
            subject_folder = self.data_path / f"Subject_{p_id}"
            self.__writeCSV(subject_folder)
        # set locale
        locale.setlocale(locale.LC_ALL, "de_DE")
        df = pd.DataFrame(
            {
                "Subject": [ind[0] for ind in index],
                "Phase": [ind[1] for ind in index],
            }
        )
        if len(df) == 0:
            raise ValueError(
                "The dataset is empty. Are you sure you selected the correct folder? Current folder is: "
                f"{self.data_path}"
            )
        return df

    @property
    def phase_time(self) -> pd.DataFrame:
        """Returns Dataframe with the time of the beginning of a phase as an absolut date for"""
        if not (
            self.is_single(None)
            or (
                self.is_single(["Subject"])
                and self.groupby("Phase").shape[0] == len(self.PHASES)
            )
        ):
            raise ValueError(
                "Data can only be accessed, when there is just a single participant in the dataset."
            )

        subject_number = self.index["Subject"][0]
        phase_name = self.index["Phase"][0]
        subject_folder = self.data_path / f"Subject_{subject_number}"
        csv_path = self.__writeCSV(subject_folder)
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)
            keys, values = rows[0], rows[1]
        # return dataframe with all times
        return pd.DataFrame(
            {
                "Subject": [subject_number],
                "Phase": phase_name,
                "Time": [values[keys.index(phase_name)]],
            }
        )

    @property
    def sampling_rate_nilspod(self) -> float:
        """The sampling rate of the raw ECG recording in Hz"""
        return self.SAMPLING_RATE_NILSPOD

    @property
    def sampling_rate_biopac(self) -> float:
        """The sampling rate of the Biopac recording in Hz"""
        return self.SAMPLING_RATE_BIOPAC

    @property
    def ecg(self) -> pd.DataFrame:
        """The raw ECG data of a participant's recording.
        The dataframe contains a single column called "ecg".
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        # Check if there is only a single participant in the dataset
        if self.is_single("Phase"):
            subject_id = self.index["Subject"][0]
            phase = self.index["Phase"][0]  # unique values in Phase as list
            return self._load_ecg(subject_id, phase)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            subject_id = self.index["Subject"][0]
            return self._load_ecg(subject_id=subject_id, phase=None)

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def _load_ecg(self, subject_id: str, phase: str) -> pd.DataFrame:

        """Return Dataframe of the requested phase or the whole respiratory_signal if phase is None"""
        # Reconstruct the ecg file path based on the data index
        nilspod_file = (
            self.data_path
            / f"Subject_{subject_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{subject_id}.bin"
        )
        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(
            nilspod_file, datastreams="ecg", handle_counter_inconsistency="ignore"
        )
        # ecg needs no calibration
        # shift the timeindex
        timedelta = self.__get_time_diff()
        df.index = df.index - timedelta
        # cut dataframe to only get one timeslot
        return self.__cut_df(df=df, phase=phase, subject_id=subject_id)

    @property
    def gyr(self) -> pd.DataFrame:
        """The gyrometer data of a participant's recording.

        The dataframe contains three columns named gyr_x, gyr_y and gyr_z.
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        subject_id = self.index["Subject"][0]
        if self.is_single(["Phase"]):
            phase = self.index["Phase"][0]  # unique values in Phase as list
            return self._load_gyr(subject_id, phase)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return self._load_gyr(subject_id=subject_id, phase=None)

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def _load_gyr(self, subject_id: str, phase: str = None):
        # Reconstruct the ecg file path based on the data index
        ecg_file = (
            self.data_path
            / f"Subject_{subject_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{subject_id}.bin"
        )
        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(
            ecg_file, handle_counter_inconsistency="warm"
        )
        # Calibrate the IMU data
        loaded_calibration = load_calibration_info(
            "/Users/Philipp/git/ba_resp_dataset/calibration/calibration_parameters.json",
            file_type="json",
        )
        df = loaded_calibration.calibrate_df(df=df, gyr_unit="deg/s", acc_unit="m/s^2")
        # select only gyr coloums
        df = df[["gyr_x", "gyr_y", "gyr_z"]]
        # shift the timeindex
        timedelta = self.__get_time_diff()
        df.index = df.index - timedelta
        # cut dataframe to only get one timeslot
        return self.__cut_df(df=df, phase=phase, subject_id=subject_id)

    @property
    def acc(self) -> pd.DataFrame:
        """The accelerometer data of a participant's recording.

        The dataframe contains three columns acc_x,_y and _z "
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        subject_id = self.index["Subject"][0]
        if self.is_single("Phase"):
            phase = self.index["Phase"][0]  # unique values in Phase as list
            return self._load_acc(subject_id, phase)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return self._load_acc(subject_id=subject_id, phase=None)

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def _load_acc(self, subject_id: str, phase: str = None):
        # Reconstruct the ecg file path based on the data index

        nilspod_file = (
            self.data_path
            / f"Subject_{subject_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{subject_id}.bin"
        )
        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(
            nilspod_file, handle_counter_inconsistency="ignore"
        )
        # Calibrate the IMU data
        loaded_calibration = load_calibration_info(
            "/Users/Philipp/git/ba_resp_dataset/calibration/calibration_parameters.json",
            file_type="json",
        )
        df = loaded_calibration.calibrate_df(df=df, acc_unit="m/s^2", gyr_unit="deg/s")
        # select only acc coloums
        df = df[["acc_x", "acc_y", "acc_z"]]
        # shift the timeindex
        timedelta = self.__get_time_diff()
        df.index = df.index - timedelta
        # cut dataframe to only get one timeslot
        return self.__cut_df(df=df, phase=phase, subject_id=subject_id)

    @property
    def nilspod(self) -> pd.DataFrame:
        """
        The whole Portabiles data of a participant's recording.
        """
        # Reconstruct the ecg file path based on the data index
        p_id = self.index["Subject"][0]

        ecg_file = (
            self.data_path
            / f"Subject_{p_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{p_id}.bin"
        )

        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(
            ecg_file, handle_counter_inconsistency="ignore"
        )

        # Calibrate the IMU data
        loaded_calibration = load_calibration_info(
            "/Users/Philipp/git/ba_resp_dataset/calibration/calibration_parameters.json",
            file_type="json",
        )
        df = loaded_calibration.calibrate_df(df=df, acc_unit="m/s^2", gyr_unit="deg/s")

        if self.is_single(None):
            phase = self.index["Phase"][0]  # unique values in Phase as list
            return self.__cut_df(df, phase, p_id)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return df

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    @property
    def biopac_ecg(self) -> pd.DataFrame:
        """
        ECG Values from Biopac recording
        """
        p_id = self.index["Subject"][0]
        txt_file = (
            self.data_path
            / f"Subject_{p_id}"
            / "raw"
            / "Biopac"
            / f"Subject_{p_id}.txt"
        )
        df = pd.read_csv(
            txt_file,
            delimiter="\t",
            skiprows=[i for i in range(0, 11) if i != 9],
        )
        # Get Begining of the Recording
        start_date = self.__get_phase_time(self.PHASES[0], p_id)

        df["min"] = pd.to_timedelta(df["min"], unit="minutes") + start_date

        df = (
            df.dropna(axis=1, how="all")
            .drop(["CH2", "CH4"], axis=1)
            .set_index("min", drop=True)
        )
        # Determine what to return
        if self.is_single("Phase"):
            phase = self.index["Phase"][0]  # unique values in Phase as list
            return self.__cut_df(df=df, phase=phase, subject_id=p_id)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return self.__cut_df(df=df, phase=None, subject_id=p_id)

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    @property
    def respiration(self) -> pd.DataFrame:
        """
        Respiration values from Biopac recording normalized between 0 and 1
        """
        p_id = self.index["Subject"][0]
        txt_file = (
            self.data_path
            / f"Subject_{p_id}"
            / "raw"
            / "Biopac"
            / f"Subject_{p_id}.txt"
        )

        df = pd.read_csv(
            txt_file,
            delimiter="\t",
            skiprows=[i for i in range(0, 11) if i != 9],
        )
        # Get Begining of the Recording
        start_date = self.__get_phase_time(self.PHASES[0], p_id)
        # Convert time from txt-file to Timedelta and add timedelta on starting time (date)
        df["min"] = pd.to_timedelta(df["min"], unit="minutes") + start_date

        # Drop CH1(Respiration) and set time as an index
        df = (
            df.dropna(axis=1, how="all")
            .drop(["CH1", "CH4"], axis=1)
            .set_index("min", drop=True)
        )
        # Determine what to return
        if self.is_single("Phase"):
            phase = self.index["Phase"][0]
            return self.__cut_df(df=df, phase=phase, subject_id=p_id)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return self.__cut_df(df=df, phase=None, subject_id=p_id)

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    @property
    def biopac_sync(self) -> pd.DataFrame:
        """
        Returns the synchronisation respiratory_signal of the Biopac device
        """
        p_id = self.index["Subject"][0]
        txt_file = (
            self.data_path
            / f"Subject_{p_id}"
            / "raw"
            / "Biopac"
            / f"Subject_{p_id}.txt"
        )

        df = pd.read_csv(
            txt_file,
            delimiter="\t",
            skiprows=[i for i in range(0, 11) if i != 9],
        )
        # Get Begining of the Recording
        start_date = self.__get_phase_time(self.PHASES[0], p_id)
        # Convert time from txt-file to Timedelta and add timedelta on starting time (date)
        df["min"] = pd.to_timedelta(df["min"], unit="minutes") + start_date
        # Drop CH1(Respiration) and set time as an index
        df = (
            df.dropna(axis=1, how="all")
            .drop(["CH1", "CH2"], axis=1)
            .set_index("min", drop=True)
        )

        # Determine what to return
        if self.is_single(None):
            # Get phase as String
            phase = self.index["Phase"][0]
            return df

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return df
        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    @property
    def nilspod_sync(self) -> pd.DataFrame:
        """
        The sensor data of the SyncPod
        Note: Cant be cuted to phases
        """
        # Reconstruct the ecg file path based on the data index
        p_id = self.index["Subject"][0]

        sync_file = (
            self.data_path
            / f"Subject_{p_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{p_id}_Sync.bin"
        )

        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(
            file_path=sync_file, datastreams=["analog"]
        )

        if self.is_single(None):
            return df

        elif self.is_single(groupby_cols="Subject") and self.groupby("Phase").shape[
            0
        ] == len(self.PHASES):
            return df

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def __writeCSV(self, path: Path) -> Path:
        """Writes a csv-file for a subjects phase times
        expects the foldername of the Subject as argument
        """
        # path of jcq file
        jcq_path = path / "raw" / "Biopac" / f"Subject_{path.__str__()[-2:]}.jcq"
        csv_path = path / "raw" / f"phase_times_{path.__str__()[-2:]}.csv"
        if csv_path.is_file():
            return csv_path

        with open(jcq_path, "r", errors="surrogateescape") as reader, open(
            csv_path, "w", encoding="UTF8"
        ) as csv_file:
            jcq_content = reader.read()
            # extract all dates
            regex = "[A-Z][a-z] .{2,3} \d{1,2} \d\d:\d\d:\d\d \d{4}"
            dates = re.findall(
                pattern=regex,
                string=jcq_content,
            )
            # Write CSV with keys (self.PHASES) and values (dates)
            writer = csv.writer(csv_file)
            writer.writerow(self.PHASES)
            writer.writerow(dates)
        return csv_path

    def __get_phase_time(self, phase: str, subject: str) -> datetime:
        subject_folder = self.data_path / f"Subject_{subject}"
        csv_path = (
            subject_folder / "raw" / f"phase_times_{subject_folder.__str__()[-2:]}.csv"
        )

        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            rows = list(reader)
            phase_times = {phase: time for (phase, time) in zip(rows[0], rows[1])}

        return pd.to_datetime(
            phase_times[phase][3:], format="%b %d %H:%M:%S %Y"
        ).tz_localize("Europe/Berlin")

    def __cut_df(self, df: pd.DataFrame, phase: str, subject_id: str) -> pd.DataFrame:
        """
        Cut df with time index to only the rows within the phase time interval
        if phase is None than the whole Signal should be returned beginning from the first official Phase from self.PHASES[0]
        """
        if phase is None:
            sync_time = self.__get_sync()
            # print("Dataframe is cutted {}".format(sync_time))
            # print("First index of Dataframe is {}".format(df.index[0]))
            return df.loc[sync_time:]

        next_phase = self.PHASES[(self.PHASES.index(phase) + 1) % len(self.PHASES)]
        if next_phase == "Initialization":
            next_phase = "Baseline 5"
        begin_time = self.__get_phase_time(subject=subject_id, phase=phase)
        # if self.PHASES.index(next_phase) == 0:
        #     # last phase
        #     return df.loc[begin_time:]
        # else:
        end_time = self.__get_phase_time(subject=subject_id, phase=next_phase)
        sync_time = self.__get_sync()

        if self.PHASES.index(phase) == 0:
            return df.loc[sync_time:end_time]
        return df.loc[begin_time:end_time]

    def __get_time_diff(self):
        """Find the time of the sync Signal on Biopac and the Nilspds and return their timeindex as tuple [0] => Nilspod [1] => Biopac"""
        nilspod_idx = self.nilspod_sync.idxmax()
        nilspod_time = pd.to_datetime(nilspod_idx[0])

        biopac_idx = self.biopac_sync.idxmax()
        biopac_time = pd.to_datetime(biopac_idx[0])

        if self.is_single(None):
            return nilspod_time - biopac_time

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return nilspod_time - biopac_time
        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def __get_sync(self):
        biopac_idx = self.biopac_sync.idxmax()
        sync_time = pd.to_datetime(biopac_idx[0])
        return sync_time

    def save_graphs(self, format: str = "pdf") -> list:
        if self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            # recreate Subject Folder path
            p_id = self.index["Subject"][0]
            image_folder = Path(
                self.data_path / f"Subject_{p_id}" / "processed" / "plots"
            )
            image_folder.mkdir(parents=True, exist_ok=True)
            # Create Plot of ecg
            fig = plt.figure(int(p_id.lstrip("0")) * 10 + 0)
            ax = fig.subplots()
            ax.plot(self.biopac_ecg)
            fig.savefig(image_folder / f"subject{p_id}_biopac_ecg.{format}")
            # Create Plot of respiration
            fig = plt.figure(int(p_id.lstrip("0")) * 10 + 1)
            ax = fig.subplots()
            ax.plot(self.respiration)
            fig.savefig(image_folder / f"subject{p_id}_respiration.{format}")
            # Create Plot of acceleration
            fig = plt.figure(int(p_id.lstrip("0")) * 10 + 2)
            ax = fig.subplots()
            ax.plot(self.acc)
            fig.savefig(image_folder / f"subject{p_id}_acceleration.{format}")
            # Create Plot of gyroscope
            fig = plt.figure(int(p_id.lstrip("0")) * 10 + 3)
            ax = fig.subplots()
            ax.plot(self.gyr)
            fig.savefig(image_folder / f"subject{p_id}_gyroscope.{format}")
            # Create Plot of Nilspod ECG
            fig = plt.figure(int(p_id.lstrip("0")) * 10 + 4)
            ax = fig.subplots()
            ax.plot(self.ecg)
            fig.savefig(image_folder / f"subject{p_id}_nilspod_ecg.{format}")
            return [image for image in sorted(image_folder.glob(f"*.{format}"))]
        else:
            raise ValueError("Data can only be accessed for a single participant ")

    def save_graph_phases(self, format: str = "pdf"):
        if self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            # recreate Subject Folder path
            p_id = self.index["Subject"][0]
            sensors = ["acc", "gyr", "resp", "ecg", "nils_ecg"]
            image_folder = Path(
                self.data_path / f"Subject_{p_id}" / "processed" / "plots"
            )
            for sensor in sensors:
                sensor_folder = Path(
                    self.data_path / f"Subject_{p_id}" / "processed" / "plots" / sensor
                )
                sensor_folder.mkdir(parents=True, exist_ok=True)
            # print(self.index["Phase"])
            for phase in self.iter_level("Phase"):
                phase_str = phase.index["Phase"][0]
                fig = plt.figure(
                    int(p_id.lstrip("0")) * 1000 + self.PHASES.index(phase_str) * 10 + 1
                )
                ax = fig.subplots()
                ax.plot(phase.ecg)
                fig.savefig(
                    image_folder
                    / "ecg"
                    / f"subject{p_id}_{phase_str}_biopac_ecg.{format}"
                )

                fig = plt.figure(
                    int(p_id.lstrip("0")) * 1000 + self.PHASES.index(phase_str) * 10 + 2
                )
                ax = fig.subplots()
                ax.plot(phase.respiration)
                fig.savefig(
                    image_folder
                    / "resp"
                    / f"subject{p_id}_{phase_str}_respiration.{format}"
                )

                fig = plt.figure(
                    int(p_id.lstrip("0")) * 1000 + self.PHASES.index(phase_str) * 10 + 3
                )
                ax = fig.subplots()
                ax.plot(phase.ecg)
                fig.savefig(
                    image_folder
                    / "nils_ecg"
                    / f"subject{p_id}_{phase_str}_ecg.{format}"
                )

                fig = plt.figure(
                    int(p_id.lstrip("0")) * 1000 + self.PHASES.index(phase_str) * 10 + 4
                )
                ax = fig.subplots()
                ax.plot(phase.acc)
                fig.savefig(
                    image_folder
                    / "acc"
                    / f"subject{p_id}_{phase_str}_acceleration.{format}"
                )

                fig = plt.figure(
                    int(p_id.lstrip("0")) * 1000 + self.PHASES.index(phase_str) * 10 + 5
                )
                ax = fig.subplots()
                ax.plot(phase.gyr)
                fig.savefig(
                    image_folder
                    / "gyr"
                    / f"subject{p_id}_{phase_str}_gyroscope.{format}"
                )
                plt.close("all")
            return [image for image in sorted(image_folder.glob(f"*/*.{format}"))]
        else:
            raise ValueError("Data can only be accessed for a single participant ")
