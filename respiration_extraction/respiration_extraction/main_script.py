import gc
import glob
import os

from pympler.tracker import SummaryTracker
import pandas as pd

from respiration_extraction.algorithms import *
from respiration_extraction.pipelines import *
from respiration_extraction.datasets import *
from pathlib import Path

ecg_extractions = [
    ExtractionKarlen,
    ExtractionCharlton,
    ExtractionVangent2019,
    ExtractionLindeberg,
    ExtractionSoni2019,
    ExtractionAddisonAM,
    ExtractionAddisonFM,
    ExtractionOrphandiou,
    ExtractionSarkar2015,
]
imu_extractions = [PositionalVectorExtraction, SavGolExtractionGyr, SavGolExtractionAcc]
estimations = [
    PeakDetection,
    PeakThroughDetection,
    CountOrig,
    CountAdvDetection,
    GradiantDetection,
]
fusion = [SmartFusion]

PATH = Path("/Users/Philipp/git/ba_resp_dataset/subjects/philipp_dataset")


def main():
    create_csvs()
    big_df = load_csvs()


def load_csvs():
    all_csvs = glob.glob(os.path.join(PATH, "/long/*.csv"))
    print(all_csvs)


def create_csvs():
    # Define Dataset
    dataset = BachelorRespirationDatasetRaw(data_path=PATH)
    dataset = dataset.get_subset(Subject=["12"])

    for datapoint in dataset:
        # Create csv results for all phases
        create_csv_result(datapoint)


def create_csv_result(datapoint):
    # define result Dataframe
    result_dataframe_wide = create_dataframe(datapoint)

    index_cols = [
        "Study",
        "Participant",
        "Phase",
        "Extraction",
        "Estimation",
        "Sensor",
    ]
    result_dataframe_long = pd.DataFrame(
        result_dataframe_wide.set_index(index_cols).stack()
    )
    subject_path = (
        PATH / "Subject_{}".format(datapoint.index["Subject"][0]) / "processed"
    )
    file_name = "{}.csv".format(datapoint.index["Phase"][0])
    out_dir = subject_path / "wide"
    out_dir.mkdir(exist_ok=True)
    result_dataframe_wide.to_csv(out_dir / file_name)

    out_dir = subject_path / "long"
    out_dir.mkdir(exist_ok=True)
    result_dataframe_long.to_csv(out_dir / file_name)


def create_dataframe(datapoint) -> pd.DataFrame:
    print(
        "Create Dataframe for participant {}, phase {}".format(
            datapoint.index["Subject"][0], datapoint.index["Phase"][0]
        )
    )

    result_dataframe_wide = pd.DataFrame(
        columns=[
            "Study",
            "Extraction",
            "Estimation",
            "Sensor",
            "Participant",
            "Phase",
            "RR",
            "GT",
            "Difference",
            "CC",
        ]
    )

    # iterate over biopac ecg algorithms
    for extr in ecg_extractions:
        for esti in estimations:
            # Create Objects

            extraction = extr()
            estimation = esti()
            # Create Pipeline
            pipeline = BiopacEDR(extraction=extraction, estimation=estimation)
            # run the method
            score_dict = pipeline.score(datapoint=datapoint)
            row_dict = {
                "Study": "BA_Philipp",
                "Extraction": extraction.__class__.__name__,
                "Estimation": estimation.__class__.__name__,
                "Sensor": "Biopac ECG",
                "Participant": datapoint.index["Subject"],
                "Phase": datapoint.index["Phase"],
                "RR": pipeline.respiration_rate,
                "GT": score_dict["GT"],
                "Difference": score_dict["GT"] - score_dict["RR"],
                "CC": score_dict["correlation"],
            }
            append = pd.DataFrame(row_dict)
            result_dataframe_wide = pd.concat(
                [result_dataframe_wide, append], ignore_index=True
            )

    # iterate over nilspod ecg algorithms
    for extr in ecg_extractions:
        for esti in estimations:
            # Create Objects
            extraction = extr()
            estimation = esti()
            # Create Pipeline
            pipeline = NilsPodEDR(extraction=extraction, estimation=estimation)
            # run the method
            score_dict = pipeline.score(datapoint=datapoint)
            row_dict = {
                "Study": "BA_Philipp",
                "Extraction": extraction.__class__.__name__,
                "Estimation": estimation.__class__.__name__,
                "Sensor": "NilsPod ECG",
                "Participant": datapoint.index["Subject"],
                "Phase": datapoint.index["Phase"],
                "RR": pipeline.respiration_rate,
                "GT": score_dict["GT"],
                "Difference": score_dict["GT"] - score_dict["RR"],
                "CC": score_dict["correlation"],
            }
            append = pd.DataFrame(row_dict)
            result_dataframe_wide = pd.concat(
                [result_dataframe_wide, append], ignore_index=True
            )
    for imu_extr in imu_extractions:
        for esti in estimations:
            # Create Objects
            extraction = imu_extr()
            estimation = esti()
            # Create Pipeline
            pipeline = ImuPipline(extraction, estimation)
            # Execute Pipeline
            score_dict = pipeline.score(datapoint=datapoint)
            row_dict = {
                "Study": "BA_Philipp",
                "Extraction": extraction.__class__.__name__,
                "Estimation": estimation.__class__.__name__,
                "Sensor": "NilsPod IMU",
                "Participant": datapoint.index["Subject"],
                "Phase": datapoint.index["Phase"],
                "RR": pipeline.respiration_rate,
                "GT": score_dict["GT"],
                "Difference": score_dict["GT"] - score_dict["RR"],
                "CC": score_dict["correlation"],
            }
            append = pd.DataFrame(row_dict)
            result_dataframe_wide = pd.concat(
                [result_dataframe_wide, append], ignore_index=True
            )
    return result_dataframe_wide


if __name__ == "__main__":
    create_csvs()
