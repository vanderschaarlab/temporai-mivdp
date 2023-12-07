# Based on
# https://github.com/healthylaife/MIMIC-IV-Data-Pipeline
# preprocessing/day_intervals_preproc/day_intervals_cohort.py

import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import disease_cohort


def get_visit_pts(
    mimic4_path: str,
    group_col: str,
    visit_col: str,
    admit_col: str,
    disch_col: str,
    adm_visit_col: Optional[str],
    use_admn: bool,
    disease_label: str,
    use_ICU: bool,
) -> pd.DataFrame:
    """Combines the MIMIC-IV core/patients table information with either the ``icu/icustays`` or ``core/admissions``
    data.

    Args:
        mimic4_path (str):
            Path to mimic-iv folder containing MIMIC-IV data.
        group_col (str):
            Patient identifier to group patients (normally ``"subject_id"``).
        visit_col (str):
            Visit identifier for individual patient visits (normally ``"hadm_id"`` or ``"stay_id"``).
        admit_col (str):
            Column for visit start date information (normally ``"admittime"`` or ``"intime"``).
        disch_col (str):
            Column for visit end date information (normally ``"dischtime"`` or ``"outtime"``).
        adm_visit_col (Optional[str]):
            TBC
        use_admn (bool):
            TBC
        disease_label (str):
            TBC
        use_ICU (bool):
            Describes whether to specifically look at ICU visits in ``icu/icustays`` OR look at general admissions from
            ``core/admissions``.

    Returns:
        pd.DataFrame: The processed dataframe.
    """
    visit = None  # DF containing visit information depending on using ICU or not
    if use_ICU:
        visit = pd.read_csv(
            mimic4_path + "icu/icustays.csv.gz",
            compression="gzip",
            header=0,
            index_col=None,
            parse_dates=[admit_col, disch_col],
        )
        if use_admn:
            # `icustays` doesn't have a way to identify if patient died during visit; must
            # use core/patients to remove such stay_ids for readmission labels
            pts = pd.read_csv(
                mimic4_path + "core/patients.csv.gz",
                compression="gzip",
                header=0,
                index_col=None,
                usecols=["subject_id", "dod"],
                parse_dates=["dod"],
            )
            visit = visit.merge(pts, how="inner", left_on="subject_id", right_on="subject_id")
            visit = visit.loc[(visit.dod.isna()) | (visit.dod >= visit[disch_col])]
            if len(disease_label):
                hids = disease_cohort.extract_diag_cohort(disease_label, mimic4_path)
                visit = visit[visit["hadm_id"].isin(hids["hadm_id"])]  # pyright: ignore
                print("[ READMISSION DUE TO " + disease_label + " ]")

    else:
        visit = pd.read_csv(
            mimic4_path + "core/admissions.csv.gz",
            compression="gzip",
            header=0,
            index_col=None,
            parse_dates=[admit_col, disch_col],
        )
        visit["los"] = visit[disch_col] - visit[admit_col]

        visit[admit_col] = pd.to_datetime(visit[admit_col])
        visit[disch_col] = pd.to_datetime(visit[disch_col])
        visit["los"] = pd.to_timedelta(visit[disch_col] - visit[admit_col], unit="h")
        visit["los"] = visit["los"].astype(str)
        visit[["days", "dummy", "hours"]] = visit["los"].str.split(" ", -1, expand=True)  # pyright: ignore
        visit["los"] = pd.to_numeric(visit["days"])
        visit = visit.drop(columns=["days", "dummy", "hours"])

        if use_admn:
            # Remove hospitalizations with a death; impossible for readmission for such visits:
            visit = visit.loc[visit.hospital_expire_flag == 0]
        if len(disease_label):
            hids = disease_cohort.extract_diag_cohort(disease_label, mimic4_path)
            visit = visit[visit["hadm_id"].isin(hids["hadm_id"])]  # pyright: ignore
            print("[ READMISSION DUE TO " + disease_label + " ]")

    pts = pd.read_csv(
        mimic4_path + "core/patients.csv.gz",
        compression="gzip",
        header=0,
        index_col=None,
        usecols=[
            group_col,
            "anchor_year",
            "anchor_age",
            "anchor_year_group",
            "dod",
            "gender",
        ],
    )
    # get yob to ensure a given visit is from an adult:
    pts["yob"] = pts["anchor_year"] - pts["anchor_age"]
    pts["min_valid_year"] = pts["anchor_year"] + (2019 - pts["anchor_year_group"].str.slice(start=-4).astype(int))

    # Define anchor_year corresponding to the anchor_year_group 2017-2019. This is later used to prevent consideration
    # of visits with prediction windows outside the dataset's time range (2008-2019)
    # [[group_col, visit_col, admit_col, disch_col]]
    if use_ICU:
        visit_pts = visit[[group_col, visit_col, adm_visit_col, admit_col, disch_col, "los"]].merge(
            pts[
                [
                    group_col,
                    "anchor_year",
                    "anchor_age",
                    "yob",
                    "min_valid_year",
                    "dod",
                    "gender",
                ]
            ],
            how="inner",
            left_on=group_col,
            right_on=group_col,
        )
    else:
        visit_pts = visit[[group_col, visit_col, admit_col, disch_col, "los"]].merge(
            pts[
                [
                    group_col,
                    "anchor_year",
                    "anchor_age",
                    "yob",
                    "min_valid_year",
                    "dod",
                    "gender",
                ]
            ],
            how="inner",
            left_on=group_col,
            right_on=group_col,
        )

    # Only take adult patients:
    #     visit_pts['Age']=visit_pts[admit_col].dt.year - visit_pts['yob']
    #     visit_pts = visit_pts.loc[visit_pts['Age'] >= 18]
    visit_pts["Age"] = visit_pts["anchor_age"]
    visit_pts = visit_pts.loc[visit_pts["Age"] >= 18]

    # Add Demo data
    eth = pd.read_csv(
        mimic4_path + "core/admissions.csv.gz",
        compression="gzip",
        header=0,
        usecols=["hadm_id", "insurance", "ethnicity"],
        index_col=None,
    )
    visit_pts = visit_pts.merge(eth, how="inner", left_on="hadm_id", right_on="hadm_id")

    if use_ICU:
        return visit_pts[
            [
                group_col,
                visit_col,
                adm_visit_col,
                admit_col,
                disch_col,
                "los",
                "min_valid_year",
                "dod",
                "Age",
                "gender",
                "ethnicity",
                "insurance",
            ]
        ]
    else:
        return visit_pts.dropna(subset=["min_valid_year"])[
            [
                group_col,
                visit_col,
                admit_col,
                disch_col,
                "los",
                "min_valid_year",
                "dod",
                "Age",
                "gender",
                "ethnicity",
                "insurance",
            ]
        ]


def validate_row(row, ctrl, invalid, max_year, disch_col, valid_col, gap):
    """Checks if visit's prediction window potentially extends beyond the dataset range (2008-2019).
    An 'invalid row' is NOT guaranteed to be outside the range, only potentially outside due to
    de-identification of MIMIC-IV being done through 3-year time ranges.

    To be invalid, the end of the prediction window's year must both extend beyond the maximum seen year
    for a patient AND beyond the year that corresponds to the 2017-2019 anchor year range for a patient
    """
    print("disch_col", row[disch_col])
    print(gap)
    pred_year = (row[disch_col] + gap).year
    if max_year < pred_year and pred_year > row[valid_col]:
        invalid = invalid.append(row)
    else:
        ctrl = ctrl.append(row)
    return ctrl, invalid


def partition_by_los(
    df: pd.DataFrame,
    los: int,
    group_col: str,
    admit_col: str,
    disch_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    invalid = df.loc[(df[admit_col].isna()) | (df[disch_col].isna()) | (df["los"].isna())]
    cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna()) & (~df["los"].isna())]

    # cohort=cohort.fillna(0)
    pos_cohort = cohort[cohort["los"] > los]
    neg_cohort = cohort[cohort["los"] <= los]
    neg_cohort = neg_cohort.fillna(0)
    pos_cohort = pos_cohort.fillna(0)

    pos_cohort["label"] = 1
    neg_cohort["label"] = 0

    cohort = pd.concat([pos_cohort, neg_cohort], axis=0)
    cohort = cohort.sort_values(by=[group_col, admit_col])
    # print("cohort",cohort.shape)
    print("[ LOS LABELS FINISHED ]")

    return cohort, invalid


def partition_by_readmit(
    df: pd.DataFrame,
    gap: datetime.timedelta,
    group_col: str,
    admit_col: str,
    disch_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Applies labels to individual visits according to whether or not a readmission has occurred within the specified
    ``gap`` days. For a given visit, another visit must occur within the gap window for a positive readmission label.
    The gap window starts from the ``disch_col`` time and the ``admit_col`` of subsequent visits are considered.
    """

    case = pd.DataFrame()  # hadm_ids with readmission within the gap period
    ctrl = pd.DataFrame()  # hadm_ids without readmission within the gap period
    invalid = pd.DataFrame()  # hadm_ids that are not considered in the cohort

    # Iterate through groupbys based on group_col (subject_id). Data is sorted by subject_id and admit_col (admittime)
    # to ensure that the most current hadm_id is last in a group.
    # grouped= df[[group_col, visit_col, admit_col, disch_col, valid_col]] \
    # .sort_values(by=[group_col, admit_col]).groupby(group_col)
    grouped = df.sort_values(by=[group_col, admit_col]).groupby(group_col)
    for subject, group in tqdm(grouped):  # pylint: disable=unused-variable
        max_year = group.max()[disch_col].year  # pylint: disable=unused-variable  # noqa

        if group.shape[0] <= 1:
            # ctrl, invalid = validate_row(group.iloc[0], ctrl, invalid, max_year, disch_col, valid_col, gap)
            # A group with 1 row has no readmission; goes to ctrl
            ctrl = ctrl.append(group.iloc[0])  # pyright: ignore
        else:
            for idx in range(group.shape[0] - 1):
                visit_time = group.iloc[idx][disch_col]  # For each index (a unique hadm_id), get its timestamp
                if (
                    group.loc[
                        (group[admit_col] > visit_time)
                        & (  # Readmissions must come AFTER the current timestamp
                            group[admit_col] - visit_time <= gap
                        )  # Distance between a timestamp and readmission must be within gap
                    ].shape[0]
                    >= 1
                ):  # If ANY rows meet above requirements, a readmission has occurred after that visit
                    case = case.append(group.iloc[idx])  # pyright: ignore
                else:
                    # If no readmission is found, only add to ctrl if prediction window is guaranteed to be within the
                    # time range of the dataset (2008-2019). Visits with prediction windows existing in potentially
                    # out-of-range dates (like 2018-2020) are excluded UNLESS the prediction window takes place the
                    # same year as the visit, in which case it is guaranteed to be within 2008-2019.

                    ctrl = ctrl.append(group.iloc[idx])  # pyright: ignore

            # ctrl, invalid = validate_row(group.iloc[-1], ctrl, invalid, max_year, disch_col, valid_col, gap)
            # The last hadm_id date-wise is guaranteed to have no readmission logically.
            ctrl = ctrl.append(group.iloc[-1])  # pyright: ignore
            # print(f"[ {gap.days} DAYS ] {case.shape[0] + ctrl.shape[0]}/{df.shape[0]} {visit_col}s processed")

    print("[ READMISSION LABELS FINISHED ]")
    return case, ctrl, invalid


def partition_by_mort(
    df: pd.DataFrame,
    group_col: str,
    admit_col: str,
    disch_col: str,
    death_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Applies labels to individual visits according to whether or not a death has occurred within
    the times of the specified ``admit_col`` and ``disch_col``."""

    invalid = df.loc[(df[admit_col].isna()) | (df[disch_col].isna())]

    cohort = df.loc[(~df[admit_col].isna()) & (~df[disch_col].isna())]

    # cohort["label"] = (
    #     (~cohort[death_col].isna())
    #     & (cohort[death_col] >= cohort[admit_col])
    #     & (cohort[death_col] <= cohort[disch_col])
    # )
    # cohort["label"] = cohort["label"].astype("Int32")
    # print("cohort",cohort.shape)
    # print(np.where(~cohort[death_col].isna(),1,0))
    # print(np.where(cohort.loc[death_col] >= cohort.loc[admit_col],1,0))
    # print(np.where(cohort.loc[death_col] <= cohort.loc[disch_col],1,0))

    cohort["label"] = 0
    # cohort=cohort.fillna(0)

    pos_cohort = cohort[~cohort[death_col].isna()]
    neg_cohort = cohort[cohort[death_col].isna()]
    neg_cohort = neg_cohort.fillna(0)
    pos_cohort = pos_cohort.fillna(0)
    pos_cohort[death_col] = pd.to_datetime(pos_cohort[death_col])

    pos_cohort["label"] = np.where(
        (pos_cohort[death_col] >= pos_cohort[admit_col]) & (pos_cohort[death_col] <= pos_cohort[disch_col]),
        1,
        0,
    )

    pos_cohort["label"] = pos_cohort["label"].astype("Int32")
    cohort = pd.concat([pos_cohort, neg_cohort], axis=0)
    cohort = cohort.sort_values(by=[group_col, admit_col])
    # print("cohort",cohort.shape)

    print("[ MORTALITY LABELS FINISHED ]")
    return cohort, invalid


def get_case_ctrls(
    df: pd.DataFrame,
    gap: Optional[int],
    group_col: str,
    admit_col: str,
    disch_col: str,
    death_col: str,
    use_mort=False,
    use_admn=False,
    use_los=False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handles logic for creating the labelled cohort based on arguments passed to ``extract_data()``.

    Args:
        df (pd.DataFrame):
            Dataframe with patient data.
        gap (Optional[int]):
            Specified time interval gap for readmissions.
        group_col (str):
            Patient identifier to group patients (normally ``"subject_id"``).
        admit_col (str):
            Column for visit start date information (normally ``"admittime"`` or ``"intime"``).
        disch_col (str):
            Column for visit end date information (normally ``"dischtime"`` or ``"outtime"``).
        death_col (str):
            Column indicating death for the mortality label.
        use_mort (bool, optional):
            Flag of whether to use the mortality label. Defaults to `False`.
        use_admn (bool, optional):
            Flag of whether to use the readmission label. Defaults to `False`.
        use_los (bool, optional):
            Flag of whether to use the length of stay label. Defaults to `False`.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed dataframes, ``(cohort, invalid)``.
    """

    case = None  # hadm_ids with readmission within the gap period
    ctrl = None  # hadm_ids without readmission within the gap period
    invalid = None  # hadm_ids that are not considered in the cohort

    if use_mort:
        return partition_by_mort(df, group_col, admit_col, disch_col, death_col)
    elif use_admn:
        if gap is None:
            raise ValueError("No gap specified")
        gap_td = datetime.timedelta(days=gap)
        # ^ Transform gap into a timedelta to compare with datetime columns
        case, ctrl, invalid = partition_by_readmit(df, gap_td, group_col, admit_col, disch_col)

        # case hadm_ids are labelled 1 for readmission, ctrls have a 0 label
        case["label"] = np.ones(case.shape[0]).astype(int)
        ctrl["label"] = np.zeros(ctrl.shape[0]).astype(int)

        return pd.concat([case, ctrl], axis=0), invalid
    elif use_los:
        if gap is None:
            raise ValueError("No gap specified")
        return partition_by_los(df, gap, group_col, admit_col, disch_col)

    else:
        raise ValueError("No label specified")

    # print(f"[ {gap.days} DAYS ] {invalid.shape[0]} hadm_ids are invalid")


def extract_data(
    version: str,
    use_ICU: str,
    label: str,
    time: int,
    icd_code: str,
    root_dir,
    disease_label,
    cohort_output=None,
    summary_output=None,
):
    # TODO: The docstring here.

    print(f"===========MIMIC-IV v{version}============")
    if not cohort_output:
        cohort_output = (
            "cohort_" + use_ICU.lower() + "_" + label.lower().replace(" ", "_") + "_" + str(time) + "_" + disease_label
        )
    if not summary_output:
        summary_output = (
            "summary_" + use_ICU.lower() + "_" + label.lower().replace(" ", "_") + "_" + str(time) + "_" + disease_label
        )

    if icd_code == "No Disease Filter":
        if len(disease_label):
            print(
                f"EXTRACTING FOR: | {use_ICU.upper()} | {label.upper()} DUE TO {disease_label.upper()} | {str(time)} | "
            )
        else:
            print(f"EXTRACTING FOR: | {use_ICU.upper()} | {label.upper()} | {str(time)} |")
    else:
        if len(disease_label):
            print(
                f"EXTRACTING FOR: | {use_ICU.upper()} | {label.upper()} DUE TO {disease_label.upper()} | "
                f"ADMITTED DUE TO {icd_code.upper()} | {str(time)} |"
            )
        else:
            print(
                f"EXTRACTING FOR: | {use_ICU.upper()} | {label.upper()} | "
                f"ADMITTED DUE TO {icd_code.upper()} | {str(time)} |"
            )
    # print(label)

    # cohort, invalid
    # ^ Final labelled output and df of invalid records, respectively
    # pts
    # ^ Valid patients generated by get_visit_pts based on use_ICU and label

    use_mort = label == "Mortality"  # change to boolean value
    use_admn = label == "Readmission"
    los = 0
    use_los = label == "Length of Stay"

    # print(use_mort)
    # print(use_admn)
    # print(use_los)
    if use_los:
        los = time
    ICU = use_ICU
    use_ICU_bool = use_ICU == "ICU"  # Change to boolean value
    use_disease = icd_code != "No Disease Filter"

    if use_ICU_bool:
        group_col = "subject_id"
        visit_col = "stay_id"
        admit_col = "intime"
        disch_col = "outtime"
        death_col = "dod"
        adm_visit_col = "hadm_id"
    else:
        group_col = "subject_id"
        visit_col = "hadm_id"
        admit_col = "admittime"
        disch_col = "dischtime"
        death_col = "dod"
        adm_visit_col = None

    pts = get_visit_pts(
        mimic4_path=root_dir + f"/mimiciv/{version}/",
        group_col=group_col,
        visit_col=visit_col,
        admit_col=admit_col,
        disch_col=disch_col,
        adm_visit_col=adm_visit_col,
        use_admn=use_admn,
        disease_label=disease_label,
        use_ICU=use_ICU_bool,
    )
    # print("pts",pts.head())

    # cols to be extracted from get_case_ctrls
    cols = [
        group_col,
        visit_col,
        admit_col,
        disch_col,
        "Age",
        "gender",
        "ethnicity",
        "insurance",
        "label",
    ]

    if use_mort:
        cols.append(death_col)
        cohort, invalid = get_case_ctrls(  # pylint: disable=unused-variable
            pts,
            None,
            group_col,
            admit_col,
            disch_col,
            death_col,
            use_mort=True,
            use_admn=False,
            use_los=False,
        )
    elif use_admn:
        interval = time
        cohort, invalid = get_case_ctrls(
            pts,
            interval,
            group_col,
            admit_col,
            disch_col,
            death_col,
            use_mort=False,
            use_admn=True,
            use_los=False,
        )
    elif use_los:
        cohort, invalid = get_case_ctrls(
            pts,
            los,
            group_col,
            admit_col,
            disch_col,
            death_col,
            use_mort=False,
            use_admn=False,
            use_los=True,
        )
    else:
        raise ValueError("No label specified")
    # print(cohort.head())

    if use_ICU_bool:
        cols.append(adm_visit_col)  # type: ignore
    # print(cohort.head())

    if use_disease:
        hids = disease_cohort.extract_diag_cohort(icd_code, root_dir + f"/mimiciv/{version}/")
        # print(hids.shape)
        # print(cohort.shape)
        # print(len(list(set(hids['hadm_id'].unique()).intersection(set(cohort['hadm_id'].unique())))))

        cohort = cohort[cohort["hadm_id"].isin(hids["hadm_id"])]  # pyright: ignore
        cohort_output = cohort_output + "_" + icd_code
        summary_output = summary_output + "_" + icd_code
    # print(cohort[cols].head())

    # Save output:
    cohort[cols].to_csv(
        root_dir + "/data/cohort/" + cohort_output + ".csv.gz",
        index=False,
        compression="gzip",
    )
    print("[ COHORT SUCCESSFULLY SAVED ]")

    summary = "\n".join(
        [
            f"{label} FOR {ICU} DATA",
            f"# Admission Records: {cohort.shape[0]}",
            f"# Patients: {cohort[group_col].nunique()}",
            f"# Positive cases: {cohort[cohort['label']==1].shape[0]}",
            f"# Negative cases: {cohort[cohort['label']==0].shape[0]}",
        ]
    )

    # Save basic summary of data:
    with open(f"./data/cohort/{summary_output}.txt", "w", encoding="utf8") as f:
        f.write(summary)

    print("[ SUMMARY SUCCESSFULLY SAVED ]")
    print(summary)

    return cohort_output
