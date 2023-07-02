import csv
import itertools
from pathlib import Path
import biopsykit.io.nilspod
import pandas as pd
import re
import locale
import datetime
from tpcp import Dataset
from typing import Optional, Union, List, Sequence


class RespirationDatasetRaw(Dataset):
    data_path: Path
    SAMPLING_RATE_NILSPOD: float = 256.0
    SAMPLING_RATE_BIOPAC: float = 250.0

    PHASES: list = [
        "Segment 1",
        "Baseline 1",
        "Speaking",
        "Baseline 2",
        "Standing and rest",
        "Sitting and rest",
        "Stroop test",
        "Baseline 3",
        "4-7-8 breathing",
        "Inhalation and hold",
        "Hyperventilation",
        "Baseline 4",
    ]

    def __init__(
        self,
        data_path: Path,
        *,
        groupby_cols: Optional[Union[List[str], str]] = None,
        subset_index: Optional[pd.DataFrame] = None,
    ):
        self.data_path = data_path
        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)

    def create_index(self) -> pd.DataFrame:
        participant_ids = [
            f.name.split("_")[1] for f in sorted(self.data_path.glob("Subject*/"))
        ]
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
        """The time of the beginning of a phase as an absolut date"""
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
        subject_id = self.index["Subject"][0]
        if self.is_single(None):
            phase = self.index["Phase"][0]  # unique values in Phase as list
            return self._load_ecg(subject_id, phase)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return self._load_ecg(subject_id=subject_id, phase=None)

        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def _load_ecg(self, subject_id: str, phase: str) -> pd.DataFrame:

        """Return Dataframe of the requested phase or the whole respiratory_signal if phase is None"""

        # Reconstruct the ecg file path based on the data index
        ecg_file = (
            self.data_path
            / f"Subject_{subject_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{subject_id}.bin"
        )

        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(ecg_file, datastreams="ecg")
        # cut dataframe to only get one timeslot
        if phase is None:
            return df
        else:
            # get next phase
            next_phase = self.PHASES[(self.PHASES.index(phase) + 1) % len(self.PHASES)]
            begin_time = self.__get_phase_time(subject=subject_id, phase=phase)
            if self.PHASES.index(next_phase) == 0:
                # last phase filter Time
                return df.loc[begin_time:]
            else:
                end_time = self.__get_phase_time(subject=subject_id, phase=next_phase)

                return df[begin_time:end_time]

    @property
    def gyr(self) -> pd.DataFrame:
        """The gyrometer data of a participant's recording.

        The dataframe contains three columns named gyr_x, gyr_y and gyr_z.
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        subject_id = self.index["Subject"][0]
        if self.is_single(None):
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
            ecg_file, datastreams=["gyro"]
        )
        # cut dataframe to only get one timeslot
        if phase is None:
            return df
        else:
            return self.__cut_df(df=df, subject_id=subject_id, phase=phase)

    @property
    def acc(self) -> pd.DataFrame:
        """The accelerometer data of a participant's recording.

        The dataframe contains three columns acc_x,_y and _z "
        The index values are just samples.
        You can use the sampling rate (`self.sampling_rate_hz`) to convert it into time
        """
        subject_id = self.index["Subject"][0]
        if self.is_single(None):
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

        ecg_file = (
            self.data_path
            / f"Subject_{subject_id}"
            / "raw"
            / "Portabiles"
            / f"Subject_{subject_id}.bin"
        )
        # Create and return Dataframe from file
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(ecg_file, datastreams=["acc"])
        # cut dataframe to only get one timeslot
        if phase is None:
            return df
        else:
            return self.__cut_df(df, phase, subject_id)

    @property
    def portabile(self) -> pd.DataFrame:
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
        df, _ = biopsykit.io.nilspod.load_dataset_nilspod(ecg_file)

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
    def txt_ECG(self) -> pd.DataFrame:
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
            skiprows=[i for i in range(0, 9) if i != 7],
        )
        # Get Begining of the Recording
        start_date = self.__get_phase_time(self.PHASES[0], p_id)

        df["min"] = pd.to_timedelta(df["min"], unit="minutes") + start_date

        df = (
            df.dropna(axis=1, how="all").drop("CH2", axis=1).set_index("min", drop=True)
        )
        # Determine what to return
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
    def txt_respiration(self) -> pd.DataFrame:
        """
        Respiration values from Biopac recording
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
            skiprows=[i for i in range(0, 9) if i != 7],
        )
        # Get Begining of the Recording
        start_date = self.__get_phase_time(self.PHASES[0], p_id)
        # Convert time from txt-file to Timedelta and add timedelta on starting time (date)
        df["min"] = pd.to_timedelta(df["min"], unit="minutes") + start_date

        # Drop CH1(Respiration) and set time as an index
        df = (
            df.dropna(axis=1, how="all").drop("CH1", axis=1).set_index("min", drop=True)
        )

        # Determine what to return
        if self.is_single(None):
            # Get phase as String
            phase = self.index["Phase"][0]
            return self.__cut_df(df=df, phase=phase, subject_id=p_id)

        elif self.is_single(["Subject"]) and self.groupby("Phase").shape[0] == len(
            self.PHASES
        ):
            return df
        raise ValueError(
            "Data can only be accessed for a single participant or a single phase "
            "of one single participant in the subset"
        )

    def __writeCSV(self, path: Path) -> None:
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
            regex = "[A-Z][a-z] .* \d\d \d\d:\d\d:\d\d \d{4}"
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
        """
        next_phase = self.PHASES[(self.PHASES.index(phase) + 1) % len(self.PHASES)]
        begin_time = self.__get_phase_time(subject=subject_id, phase=phase)

        if self.PHASES.index(next_phase) == 0:
            # last phase
            return df.loc[begin_time:]
        else:
            end_time = self.__get_phase_time(subject=subject_id, phase=next_phase)
            return df.loc[begin_time:end_time]
