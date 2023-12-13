"""ICU feature preprocessing module.

Based on:
https://github.com/healthylaife/MIMIC-IV-Data-Pipeline
``preprocessing/hosp_module_preproc/feature_selection_icu.py``
"""

import os
from typing import Optional, Tuple

import pandas as pd
from typing_extensions import Literal, get_args

from ...utils import icu_preprocess_util, outlier_removal, uom_conversion
from ..cohort.disease_cohort import ICD_MAP_PATH

OutDfs = Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
]


def feature_icu(
    cohort_output: str,
    root_dir: str,
    version: str,
    diag_flag: bool = True,
    out_flag: bool = True,
    chart_flag: bool = True,
    proc_flag: bool = True,
    med_flag: bool = True,
) -> OutDfs:
    """Extracts features from ICU data.

    Args:
        cohort_output (str):
            Cohort output file name.
        root_dir (str):
            Root directory of the MIMIC-IV dataset.
        version (str):
            MIMIC-IV version string, e.g. ``"v1_0"``.
        diag_flag (bool, optional):
            Whether to extract diagnosis data. Defaults to `True`.
        out_flag (bool, optional):
            Whether to extract output events data. Defaults to `True`.
        chart_flag (bool, optional):
            Whether to extract chart events data. Defaults to `True`.
        proc_flag (bool, optional):
            Whether to extract procedures data. Defaults to `True`.
        med_flag (bool, optional):
            Whether to extract medications data. Defaults to `True`.

    Returns:
        OutDfs: Output dataframes ``diag, out, chart, proc, med``, depending on the flags.
    """
    mimic_dir = os.path.join(root_dir, f"{version}")
    out_dir = os.path.join(root_dir, "data")

    out_cohort_dir = os.path.join(out_dir, "cohort")
    out_features_dir = os.path.join(out_dir, "features")
    os.makedirs(out_features_dir, exist_ok=True)

    diag = None
    out = None
    chart = None
    proc = None
    med = None

    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = icu_preprocess_util.preproc_icd_module(
            os.path.join(mimic_dir, "hosp/diagnoses_icd.csv.gz"),
            os.path.join(out_cohort_dir, f"{cohort_output}.csv.gz"),
            ICD_MAP_PATH,
            map_code_colname="diagnosis_code",
        )
        diag[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "icd_code",
                "root_icd10_convert",
                "root",
            ]
        ].to_csv(os.path.join(out_features_dir, "preproc_diag_icu.csv.gz"), compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if out_flag:
        print("[EXTRACTING OUTPUT EVENTS DATA]")
        out = icu_preprocess_util.preproc_out(
            os.path.join(mimic_dir, "icu/outputevents.csv.gz"),
            os.path.join(out_cohort_dir, f"{cohort_output}.csv.gz"),
            "charttime",
            dtypes=None,
            usecols=None,
        )
        out[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "charttime",
                "intime",
                "event_time_from_admit",
            ]
        ].to_csv(os.path.join(out_features_dir, "preproc_out_icu.csv.gz"), compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")

    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart = icu_preprocess_util.preproc_chart(
            os.path.join(mimic_dir, "icu/chartevents.csv.gz"),
            os.path.join(out_cohort_dir, f"{cohort_output}.csv.gz"),
            "charttime",
            dtypes=None,
            usecols=["stay_id", "charttime", "itemid", "valuenum", "valueuom"],
        )
        chart = uom_conversion.drop_wrong_uom(chart, 0.95)
        chart[["stay_id", "itemid", "event_time_from_admit", "valuenum"]].to_csv(
            os.path.join(out_features_dir, "preproc_chart_icu.csv.gz"),
            compression="gzip",
            index=False,
        )
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = icu_preprocess_util.preproc_proc(
            os.path.join(mimic_dir, "icu/procedureevents.csv.gz"),
            os.path.join(out_cohort_dir, f"{cohort_output}.csv.gz"),
            "starttime",
            dtypes=None,
            usecols=["stay_id", "starttime", "itemid"],
        )
        proc[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "starttime",
                "intime",
                "event_time_from_admit",
            ]
        ].to_csv(os.path.join(out_features_dir, "preproc_proc_icu.csv.gz"), compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = icu_preprocess_util.preproc_meds(
            os.path.join(mimic_dir, "icu/inputevents.csv.gz"),
            os.path.join(out_cohort_dir, f"{cohort_output}.csv.gz"),
        )
        med[
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "itemid",
                "starttime",
                "endtime",
                "start_hours_from_admit",
                "stop_hours_from_admit",
                "rate",
                "amount",
                "orderid",
            ]
        ].to_csv(os.path.join(out_features_dir, "preproc_med_icu.csv.gz"), compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    return diag, out, chart, proc, med


GroupOption = Literal[
    "both",  # Keep both ICD-9 and ICD-10 codes
    "convert",  # Convert ICD-9 to ICD-10 codes
    "convert_group",  # Convert ICD-9 to ICD-10 and group ICD-10 codes
]


def preprocess_features_icu(
    cohort_output: str,  # pylint: disable=unused-argument
    root_dir: str,
    diag_flag: bool,
    group_diag: GroupOption,
    chart_flag: bool,
    clean_chart: bool,
    impute_outlier_chart: bool,
    thresh: int,
    left_thresh: int,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Performs grouping on diagnosis data and/or outlier removal and imputation on chart events data.

    Args:
        cohort_output (str):
            Cohort output file name.
        root_dir (str):
            Root directory of the MIMIC-IV dataset.
        dia_flag (bool):
            Whether to process diagnosis data.
        group_diag (GroupOption):
            Grouping option for diagnosis data.
            ``"both"``: Keep both ICD-9 and ICD-10 codes.
            ``"convert"``: Convert ICD-9 to ICD-10 codes.
            ``"convert_group"``: Convert ICD-9 to ICD-10 and group ICD-10 codes.
            Only applicable if ``diag_flag`` is `True`.
        chart_flag (bool):
            Whether to process chart events data.
        clean_chart (bool):
            Whether to clean chart events data. Only applicable if ``chart_flag`` is `True`.
        impute_outlier_chart (bool):
            Whether to impute outliers in chart events data. Only applicable if ``chart_flag`` is `True`.
        thresh (int):
            (Right/upper) threshold for outlier removal. Only applicable if ``chart_flag`` is `True`.
        left_thresh (int):
            (Left/lower) threshold for outlier removal. Only applicable if ``chart_flag`` is `True`.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            Dataframes ``diag, chart``, depending on the flags.
    """
    if group_diag not in get_args(GroupOption):
        raise ValueError(f"Invalid group_diag option {group_diag}, expected one of {get_args(GroupOption)}")

    diag, chart = None, None

    out_dir = os.path.join(root_dir, "data")
    out_features_dir = os.path.join(out_dir, "features")

    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(os.path.join(out_features_dir, "preproc_diag_icu.csv.gz"), compression="gzip", header=0)
        if group_diag == "both":
            diag["new_icd_code"] = diag["icd_code"]
        if group_diag == "convert":
            diag["new_icd_code"] = diag["root_icd10_convert"]
        if group_diag == "convert_group":
            diag["new_icd_code"] = diag["root"]

        diag = diag[["subject_id", "hadm_id", "stay_id", "new_icd_code"]].dropna()
        print("Total number of rows", diag.shape[0])
        diag.to_csv(os.path.join(out_features_dir, "preproc_diag_icu.csv.gz"), compression="gzip", index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if chart_flag:
        if clean_chart:
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv(
                os.path.join(out_features_dir, "preproc_chart_icu.csv.gz"), compression="gzip", header=0
            )
            chart = outlier_removal.outlier_imputation(
                chart, "itemid", "valuenum", thresh, left_thresh, impute_outlier_chart
            )

            # for i in [227441, 229357, 229358, 229360]:
            #     try:
            #         maj = chart.loc[chart.itemid == i].valueuom.value_counts().index[0]
            #         chart = chart.loc[~((chart.itemid == i) & (chart.valueuom == maj))]
            #     except IndexError:
            #         print(f"{idx} not found")

            print("Total number of rows", chart.shape[0])
            chart.to_csv(
                os.path.join(out_features_dir, "preproc_chart_icu.csv.gz"),
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    return diag, chart


def generate_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression="gzip", header=0)
        freq = diag.groupby(["stay_id", "new_icd_code"]).size().reset_index(name="mean_frequency")
        freq = freq.groupby(["new_icd_code"])["mean_frequency"].mean().reset_index()
        total = diag.groupby("new_icd_code").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="new_icd_code", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/diag_summary.csv", index=False)
        summary["new_icd_code"].to_csv("./data/summary/diag_features.csv", index=False)

    if med_flag:
        med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression="gzip", header=0)
        freq = med.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = med[med["amount"] == 0].groupby("itemid").size().reset_index(name="missing_count")
        total = med.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")
        # summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/med_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/med_features.csv", index=False)

    if proc_flag:
        proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression="gzip", header=0)
        freq = proc.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = proc.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/proc_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/proc_features.csv", index=False)

    if out_flag:
        out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression="gzip", header=0)
        freq = out.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()
        total = out.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(freq, total, on="itemid", how="right")
        summary = summary.fillna(0)
        summary.to_csv("./data/summary/out_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/out_features.csv", index=False)

    if chart_flag:
        chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression="gzip", header=0)
        freq = chart.groupby(["stay_id", "itemid"]).size().reset_index(name="mean_frequency")
        freq = freq.groupby(["itemid"])["mean_frequency"].mean().reset_index()

        missing = chart[chart["valuenum"] == 0].groupby("itemid").size().reset_index(name="missing_count")
        total = chart.groupby("itemid").size().reset_index(name="total_count")
        summary = pd.merge(missing, total, on="itemid", how="right")
        summary = pd.merge(freq, summary, on="itemid", how="right")

        # summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        # summary=summary.fillna(0)
        # final.groupby('itemid')['missing_count'].sum().reset_index()
        # final.groupby('itemid')['total_count'].sum().reset_index()
        # final.groupby('itemid')['missing%'].mean().reset_index()

        summary = summary.fillna(0)
        summary.to_csv("./data/summary/chart_summary.csv", index=False)
        summary["itemid"].to_csv("./data/summary/chart_features.csv", index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")


def features_selection_icu(
    cohort_output,  # pylint: disable=unused-argument
    diag_flag,
    proc_flag,
    med_flag,
    out_flag,
    chart_flag,
    group_diag,
    group_med,
    group_proc,
    group_out,
    group_chart,
):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression="gzip", header=0)
            features = pd.read_csv("./data/summary/diag_features.csv", header=0)
            diag = diag[diag["new_icd_code"].isin(features["new_icd_code"].unique())]

            print("Total number of rows", diag.shape[0])
            diag.to_csv(
                "./data/features/preproc_diag_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if med_flag:
        if group_med:
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression="gzip", header=0)
            features = pd.read_csv("./data/summary/med_features.csv", header=0)
            med = med[med["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", med.shape[0])
            med.to_csv(
                "./data/features/preproc_med_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression="gzip", header=0)
            features = pd.read_csv("./data/summary/proc_features.csv", header=0)
            proc = proc[proc["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", proc.shape[0])
            proc.to_csv(
                "./data/features/preproc_proc_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")

    if out_flag:
        if group_out:
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression="gzip", header=0)
            features = pd.read_csv("./data/summary/out_features.csv", header=0)
            out = out[out["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", out.shape[0])
            out.to_csv(
                "./data/features/preproc_out_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")

    if chart_flag:
        if group_chart:
            print("[FEATURE SELECTION CHART EVENTS DATA]")

            chart = pd.read_csv(
                "./data/features/preproc_chart_icu.csv.gz",
                compression="gzip",
                header=0,
                index_col=None,
            )

            features = pd.read_csv("./data/summary/chart_features.csv", header=0)
            chart = chart[chart["itemid"].isin(features["itemid"].unique())]
            print("Total number of rows", chart.shape[0])
            chart.to_csv(
                "./data/features/preproc_chart_icu.csv.gz",
                compression="gzip",
                index=False,
            )
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
