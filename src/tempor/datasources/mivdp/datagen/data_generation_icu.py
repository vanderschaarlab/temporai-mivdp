import os
import pickle  # nosec: B403
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing_extensions import Literal

from ..utils.common import pd_v2_compat_append

ImputeOption = Literal["Mean", "Median", False]


class ICUDataGenerator:
    def __init__(
        self,
        cohort_output: str,
        root_dir: str,
        if_mort: bool,
        if_admn: bool,
        if_los: bool,
        feat_cond: bool,
        feat_proc: bool,
        feat_out: bool,
        feat_chart: bool,
        feat_med: bool,
        impute: ImputeOption,
        include_time: int = 24,
        bucket: int = 1,
        predW: int = 6,
        silence_warnings: bool = True,
    ) -> None:
        """The data generator object that handles the final data processing aspects of the pipeline.

        Determines how to process and represent the time-series data.
        - You will choose the length of time-series data you want to include for this study (``include_time``).
        - You will select the ``bucket`` size which tells in what size time windows you want to divide your time-series.
            For example, if you select a ``2`` bucket size, it wil aggregate data for every 2 hours and a time-series
            of length 24 hours will be represented as time-series with 12 time-windows >where data for every 2 hours is
            aggregated from original raw time-series.
        - You can also choose if you want to ``impute`` chart values. The imputation will be done by froward fill and
            mean or median imputation. Values will be forward fill first and if no value exists for that admission we
            will use mean or median value for the patient.

        Args:
            cohort_output (str):
                Cohort output file name.
            root_dir (str):
                Root directory of the MIMIC-IV dataset.
            if_mort (bool):
                Whether the mortality task (target) is selected.
            if_admn (bool):
                Whether the readmission task (target) is selected.
            if_los (bool):
                Whether the length of stay task (target) is selected.
            feat_cond (bool):
                Whether the diagnosis features are selected.
            feat_proc (bool):
                Whether the procedure features are selected.
            feat_out (bool):
                Whether the output event features are selected.
            feat_chart (bool):
                Whether the chart features are selected.
            feat_med (bool):
                Whether the medication features are selected.
            impute (ImputeOption):
                The imputation method to use for missing values. One of ``"Mean"``, ``"Median"``, or `False`.
            include_time (int, optional):
                Number of timesteps to include. Defaults to ``24``.
            bucket (int, optional):
                Time bucket size (in hours). Defaults to ``1``.
            predW (int, optional):
                Applicable to mortality task only - the mortality prediction window. Defaults to ``6``.
            silence_warnings (bool, optional):
                Whether to silence warnings. Defaults to ``True``.
        """

        # Handle directories.
        out_dir = os.path.join(root_dir, "data")
        self._out_cohort_dir = os.path.join(out_dir, "cohort")
        self._out_features_dir = os.path.join(out_dir, "features")
        self._out_dict_dir = os.path.join(out_dir, "dict")
        self._out_csv_dir = os.path.join(out_dir, "csv")
        os.makedirs(self._out_dict_dir, exist_ok=True)
        os.makedirs(self._out_csv_dir, exist_ok=True)

        (
            self.feat_cond,
            self.feat_proc,
            self.feat_out,
            self.feat_chart,
            self.feat_med,
        ) = (feat_cond, feat_proc, feat_out, feat_chart, feat_med)
        self.cohort_output = cohort_output
        self.impute = impute

        # Initialize for typing clarity.
        self.data: pd.DataFrame
        self.cond: pd.DataFrame
        self.cond_per_adm: int
        self.proc: pd.DataFrame
        self.out: pd.DataFrame
        self.chart: pd.DataFrame
        self.meds: pd.DataFrame
        self.chart_vocab_n: int
        self.age_vocab_n: int
        self.eth_vocab_n: int
        self.ins_vocab_n: int
        self.med_vocab_n: int
        self.proc_vocab_n: int
        self.out_vocab_n: int
        self.cond_vocab_n: int
        self.dataChartDic: Dict
        self.chartVocab: List
        self.metaDic: Dict
        self.dataDic: Dict
        self.hadmDic: Any
        self.ethVocab: List
        self.ageVocab: List
        self.insVocab: List
        self.medVocab: List
        self.outVocab: List
        self.condVocab: List
        self.procVocab: List
        self.los: int
        self.hids: np.ndarray
        self.med_per_adm: Any
        self.medlength_per_adm: Any
        self.proc_per_adm: Any
        self.proclength_per_adm: Any
        self.out_per_adm: Any
        self.outlength_per_adm: Any
        self.chart_per_adm: Any
        self.chartlength_per_adm: Any
        # ------------------------------

        if silence_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._init(
                    if_mort=if_mort,
                    if_admn=if_admn,
                    if_los=if_los,
                    include_time=include_time,
                    bucket=bucket,
                    predW=predW,
                )
        else:
            self._init(
                if_mort=if_mort,
                if_admn=if_admn,
                if_los=if_los,
                include_time=include_time,
                bucket=bucket,
                predW=predW,
            )

    def _init(
        self,
        if_mort: bool,
        if_admn: bool,
        if_los: bool,
        include_time: int = 24,
        bucket: int = 1,
        predW: int = 6,
    ) -> None:
        self.data = self.generate_adm()
        print("[ READ COHORT ]")

        self.generate_feat()
        print("[ READ ALL FEATURES ]")

        if if_mort:
            self.mortality_length(include_time, predW)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_admn:
            self.readmission_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_los:
            self.los_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")

        self.smooth_meds(bucket)
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")

    def generate_feat(self):
        if self.feat_cond:
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if self.feat_proc:
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if self.feat_out:
            print("[ ======READING OUT EVENTS ]")
            self.generate_out()
        if self.feat_chart:
            print("[ ======READING CHART EVENTS ]")
            self.generate_chart()
        if self.feat_med:
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()

    def generate_adm(self):
        data = pd.read_csv(
            os.path.join(self._out_cohort_dir, f"{self.cohort_output}.csv.gz"),
            compression="gzip",
            header=0,
            index_col=None,
        )
        data["intime"] = pd.to_datetime(data["intime"])
        data["outtime"] = pd.to_datetime(data["outtime"])
        data["los"] = pd.to_timedelta(data["outtime"] - data["intime"], unit="h")
        data["los"] = data["los"].astype(str)
        data[["days", "dummy", "hours"]] = data["los"].str.split(" ", n=-1, expand=True)
        data[["hours", "min", "sec"]] = data["hours"].str.split(":", n=-1, expand=True)
        data["los"] = pd.to_numeric(data["days"]) * 24 + pd.to_numeric(data["hours"])
        data = data.drop(columns=["days", "dummy", "hours", "min", "sec"])
        data = data[data["los"] > 0]
        data["Age"] = data["Age"].astype(int)
        # print(data.head())
        # print(data.shape)
        return data

    def generate_cond(self):
        cond = pd.read_csv(
            os.path.join(self._out_features_dir, "preproc_diag_icu.csv.gz"),
            compression="gzip",
            header=0,
            index_col=None,
        )
        cond = cond[cond["stay_id"].isin(self.data["stay_id"])]
        cond_per_adm = cond.groupby("stay_id").size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm

    def generate_proc(self):
        proc = pd.read_csv(
            os.path.join(self._out_features_dir, "preproc_proc_icu.csv.gz"),
            compression="gzip",
            header=0,
            index_col=None,
        )
        proc = proc[proc["stay_id"].isin(self.data["stay_id"])]
        proc[
            [
                "start_days",
                "dummy",
                "start_hours",
            ]
        ] = proc[
            "event_time_from_admit"
        ].str.split(" ", n=-1, expand=True)
        proc[["start_hours", "min", "sec"]] = proc["start_hours"].str.split(":", n=-1, expand=True)
        proc["start_time"] = pd.to_numeric(proc["start_days"]) * 24 + pd.to_numeric(proc["start_hours"])
        proc = proc.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
        proc = proc[proc["start_time"] >= 0]

        # Remove where event time is after discharge time
        proc = pd.merge(proc, self.data[["stay_id", "los"]], on="stay_id", how="left")
        proc["sanity"] = proc["los"] - proc["start_time"]
        proc = proc[proc["sanity"] > 0]
        del proc["sanity"]

        self.proc = proc

    def generate_out(self):
        out = pd.read_csv(
            os.path.join(self._out_features_dir, "preproc_out_icu.csv.gz"),
            compression="gzip",
            header=0,
            index_col=None,
        )
        out = out[out["stay_id"].isin(self.data["stay_id"])]
        out[
            [
                "start_days",
                "dummy",
                "start_hours",
            ]
        ] = out[
            "event_time_from_admit"
        ].str.split(" ", n=-1, expand=True)
        out[["start_hours", "min", "sec"]] = out["start_hours"].str.split(":", n=-1, expand=True)
        out["start_time"] = pd.to_numeric(out["start_days"]) * 24 + pd.to_numeric(out["start_hours"])
        out = out.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
        out = out[out["start_time"] >= 0]

        # Remove where event time is after discharge time
        out = pd.merge(out, self.data[["stay_id", "los"]], on="stay_id", how="left")
        out["sanity"] = out["los"] - out["start_time"]
        out = out[out["sanity"] > 0]
        del out["sanity"]

        self.out = out

    def generate_chart(self):
        chunksize = 5000000
        final = pd.DataFrame()
        for chart in tqdm(
            pd.read_csv(
                os.path.join(self._out_features_dir, "preproc_chart_icu.csv.gz"),
                compression="gzip",
                header=0,
                index_col=None,
                chunksize=chunksize,
            )
        ):
            chart = chart[chart["stay_id"].isin(self.data["stay_id"])]
            chart[["start_days", "dummy", "start_hours"]] = chart["event_time_from_admit"].str.split(
                " ", n=-1, expand=True
            )
            chart[["start_hours", "min", "sec"]] = chart["start_hours"].str.split(":", n=-1, expand=True)
            chart["start_time"] = pd.to_numeric(chart["start_days"]) * 24 + pd.to_numeric(chart["start_hours"])
            chart = chart.drop(
                columns=[
                    "start_days",
                    "dummy",
                    "start_hours",
                    "min",
                    "sec",
                    "event_time_from_admit",
                ]
            )
            chart = chart[chart["start_time"] >= 0]

            # Remove where event time is after discharge time
            chart = pd.merge(chart, self.data[["stay_id", "los"]], on="stay_id", how="left")
            chart["sanity"] = chart["los"] - chart["start_time"]
            chart = chart[chart["sanity"] > 0]
            del chart["sanity"]
            del chart["los"]

            if final.empty:
                final = chart
            else:
                final = pd_v2_compat_append(final, chart, ignore_index=True)

        self.chart = final

    def generate_meds(self):
        meds = pd.read_csv(
            os.path.join(self._out_features_dir, "preproc_med_icu.csv.gz"),
            compression="gzip",
            header=0,
            index_col=None,
        )
        meds[
            [
                "start_days",
                "dummy",
                "start_hours",
            ]
        ] = meds[
            "start_hours_from_admit"
        ].str.split(" ", n=-1, expand=True)
        meds[["start_hours", "min", "sec"]] = meds["start_hours"].str.split(":", n=-1, expand=True)
        meds["start_time"] = pd.to_numeric(meds["start_days"]) * 24 + pd.to_numeric(meds["start_hours"])
        meds[
            [
                "start_days",
                "dummy",
                "start_hours",
            ]
        ] = meds[
            "stop_hours_from_admit"
        ].str.split(" ", n=-1, expand=True)
        meds[["start_hours", "min", "sec"]] = meds["start_hours"].str.split(":", n=-1, expand=True)
        meds["stop_time"] = pd.to_numeric(meds["start_days"]) * 24 + pd.to_numeric(meds["start_hours"])
        meds = meds.drop(columns=["start_days", "dummy", "start_hours", "min", "sec"])
        # Sanity check
        meds["sanity"] = meds["stop_time"] - meds["start_time"]
        meds = meds[meds["sanity"] > 0]
        del meds["sanity"]
        # Select hadm_id as in main file
        meds = meds[meds["stay_id"].isin(self.data["stay_id"])]
        meds = pd.merge(meds, self.data[["stay_id", "los"]], on="stay_id", how="left")

        # Remove where start time is after end of visit
        meds["sanity"] = meds["los"] - meds["start_time"]
        meds = meds[meds["sanity"] > 0]
        del meds["sanity"]
        # Any stop_time after end of visit is set at end of visit
        meds.loc[meds["stop_time"] > meds["los"], "stop_time"] = meds.loc[meds["stop_time"] > meds["los"], "los"]
        del meds["los"]

        meds["rate"] = meds["rate"].apply(pd.to_numeric, errors="coerce")
        meds["amount"] = meds["amount"].apply(pd.to_numeric, errors="coerce")

        self.meds = meds

    def mortality_length(self, include_time, predW):
        print("include_time", include_time)
        self.los = include_time
        self.data = self.data[(self.data["los"] >= include_time + predW)]
        self.hids = self.data["stay_id"].unique()

        if self.feat_cond:
            self.cond = self.cond[self.cond["stay_id"].isin(self.data["stay_id"])]

        self.data["los"] = include_time

        # --- Make equal length input time series and remove data for pred window if needed. ---

        # MEDS
        if self.feat_med:
            self.meds = self.meds[self.meds["stay_id"].isin(self.data["stay_id"])]
            self.meds = self.meds[self.meds["start_time"] <= include_time]
            self.meds.loc[self.meds.stop_time > include_time, "stop_time"] = include_time

        # PROCS
        if self.feat_proc:
            self.proc = self.proc[self.proc["stay_id"].isin(self.data["stay_id"])]
            self.proc = self.proc[self.proc["start_time"] <= include_time]

        # OUT
        if self.feat_out:
            self.out = self.out[self.out["stay_id"].isin(self.data["stay_id"])]
            self.out = self.out[self.out["start_time"] <= include_time]

        # CHART
        if self.feat_chart:
            self.chart = self.chart[self.chart["stay_id"].isin(self.data["stay_id"])]
            self.chart = self.chart[self.chart["start_time"] <= include_time]

    def los_length(self, include_time):
        print("include_time", include_time)
        self.los = include_time
        self.data = self.data[(self.data["los"] >= include_time)]
        self.hids = self.data["stay_id"].unique()

        if self.feat_cond:
            self.cond = self.cond[self.cond["stay_id"].isin(self.data["stay_id"])]

        self.data["los"] = include_time

        # --- Make equal length input time series and remove data for pred window if needed ---

        # MEDS
        if self.feat_med:
            self.meds = self.meds[self.meds["stay_id"].isin(self.data["stay_id"])]
            self.meds = self.meds[self.meds["start_time"] <= include_time]
            self.meds.loc[self.meds.stop_time > include_time, "stop_time"] = include_time

        # PROCS
        if self.feat_proc:
            self.proc = self.proc[self.proc["stay_id"].isin(self.data["stay_id"])]
            self.proc = self.proc[self.proc["start_time"] <= include_time]

        # OUT
        if self.feat_out:
            self.out = self.out[self.out["stay_id"].isin(self.data["stay_id"])]
            self.out = self.out[self.out["start_time"] <= include_time]

        # CHART
        if self.feat_chart:
            self.chart = self.chart[self.chart["stay_id"].isin(self.data["stay_id"])]
            self.chart = self.chart[self.chart["start_time"] <= include_time]

    def readmission_length(self, include_time):
        self.los = include_time
        self.data = self.data[(self.data["los"] >= include_time)]
        self.hids = self.data["stay_id"].unique()

        if self.feat_cond:
            self.cond = self.cond[self.cond["stay_id"].isin(self.data["stay_id"])]
        self.data["select_time"] = self.data["los"] - include_time
        self.data["los"] = include_time

        # --- Make equal length input time series and remove data for pred window if needed. ---

        # MEDS
        if self.feat_med:
            self.meds = self.meds[self.meds["stay_id"].isin(self.data["stay_id"])]
            self.meds = pd.merge(
                self.meds,
                self.data[["stay_id", "select_time"]],
                on="stay_id",
                how="left",
            )
            self.meds["stop_time"] = self.meds["stop_time"] - self.meds["select_time"]
            self.meds["start_time"] = self.meds["start_time"] - self.meds["select_time"]
            self.meds = self.meds[self.meds["stop_time"] >= 0]
            self.meds.loc[self.meds.start_time < 0, "start_time"] = 0

        # PROCS
        if self.feat_proc:
            self.proc = self.proc[self.proc["stay_id"].isin(self.data["stay_id"])]
            self.proc = pd.merge(
                self.proc,
                self.data[["stay_id", "select_time"]],
                on="stay_id",
                how="left",
            )
            self.proc["start_time"] = self.proc["start_time"] - self.proc["select_time"]
            self.proc = self.proc[self.proc["start_time"] >= 0]

        # OUT
        if self.feat_out:
            self.out = self.out[self.out["stay_id"].isin(self.data["stay_id"])]
            self.out = pd.merge(
                self.out,
                self.data[["stay_id", "select_time"]],
                on="stay_id",
                how="left",
            )
            self.out["start_time"] = self.out["start_time"] - self.out["select_time"]
            self.out = self.out[self.out["start_time"] >= 0]

        # CHART
        if self.feat_chart:
            self.chart = self.chart[self.chart["stay_id"].isin(self.data["stay_id"])]
            self.chart = pd.merge(
                self.chart,
                self.data[["stay_id", "select_time"]],
                on="stay_id",
                how="left",
            )
            self.chart["start_time"] = self.chart["start_time"] - self.chart["select_time"]
            self.chart = self.chart[self.chart["start_time"] >= 0]

    def smooth_meds(self, bucket):
        final_meds = pd.DataFrame()
        final_proc = pd.DataFrame()
        final_out = pd.DataFrame()
        final_chart = pd.DataFrame()

        if self.feat_med:
            self.meds = self.meds.sort_values(by=["start_time"])
        if self.feat_proc:
            self.proc = self.proc.sort_values(by=["start_time"])
        if self.feat_out:
            self.out = self.out.sort_values(by=["start_time"])
        if self.feat_chart:
            self.chart = self.chart.sort_values(by=["start_time"])

        t = 0
        for i in tqdm(range(0, self.los, bucket)):
            # MEDS
            if self.feat_med:
                sub_meds = (
                    self.meds[(self.meds["start_time"] >= i) & (self.meds["start_time"] < i + bucket)]
                    .groupby(["stay_id", "itemid", "orderid"])
                    .agg(
                        {
                            "stop_time": "max",
                            "subject_id": "max",
                            "rate": np.nanmean,
                            "amount": np.nanmean,
                        }
                    )
                )
                sub_meds = sub_meds.reset_index()
                sub_meds["start_time"] = t
                sub_meds["stop_time"] = sub_meds["stop_time"] / bucket
                if final_meds.empty:
                    final_meds = sub_meds
                else:
                    final_meds = pd_v2_compat_append(final_meds, sub_meds)

            # PROC
            if self.feat_proc:
                sub_proc = (
                    self.proc[(self.proc["start_time"] >= i) & (self.proc["start_time"] < i + bucket)]
                    .groupby(["stay_id", "itemid"])
                    .agg({"subject_id": "max"})
                )
                sub_proc = sub_proc.reset_index()
                sub_proc["start_time"] = t
                if final_proc.empty:
                    final_proc = sub_proc
                else:
                    final_proc = pd_v2_compat_append(final_proc, sub_proc)

            # OUT
            if self.feat_out:
                sub_out = (
                    self.out[(self.out["start_time"] >= i) & (self.out["start_time"] < i + bucket)]
                    .groupby(["stay_id", "itemid"])
                    .agg({"subject_id": "max"})
                )
                sub_out = sub_out.reset_index()
                sub_out["start_time"] = t
                if final_out.empty:
                    final_out = sub_out
                else:
                    final_out = pd_v2_compat_append(final_out, sub_out)

            # CHART
            if self.feat_chart:
                sub_chart = (
                    self.chart[(self.chart["start_time"] >= i) & (self.chart["start_time"] < i + bucket)]
                    .groupby(["stay_id", "itemid"])
                    .agg({"valuenum": np.nanmean})
                )
                sub_chart = sub_chart.reset_index()
                sub_chart["start_time"] = t
                if final_chart.empty:
                    final_chart = sub_chart
                else:
                    final_chart = pd_v2_compat_append(final_chart, sub_chart)

            t = t + 1
        print("bucket", bucket)
        los = int(self.los / bucket)

        # MEDS
        if self.feat_med:
            f2_meds = final_meds.groupby(["stay_id", "itemid", "orderid"]).size()
            self.med_per_adm = f2_meds.groupby("stay_id").sum().reset_index()[0].max()
            self.medlength_per_adm = final_meds.groupby("stay_id").size().max()

        # PROC
        if self.feat_proc:
            f2_proc = final_proc.groupby(["stay_id", "itemid"]).size()
            self.proc_per_adm = f2_proc.groupby("stay_id").sum().reset_index()[0].max()
            self.proclength_per_adm = final_proc.groupby("stay_id").size().max()

        # OUT
        if self.feat_out:
            f2_out = final_out.groupby(["stay_id", "itemid"]).size()
            self.out_per_adm = f2_out.groupby("stay_id").sum().reset_index()[0].max()
            self.outlength_per_adm = final_out.groupby("stay_id").size().max()

        # chart
        if self.feat_chart:
            f2_chart = final_chart.groupby(["stay_id", "itemid"]).size()
            self.chart_per_adm = f2_chart.groupby("stay_id").sum().reset_index()[0].max()
            self.chartlength_per_adm = final_chart.groupby("stay_id").size().max()

        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")
        # CREATE DICT
        # if(self.feat_chart):
        #     self.create_chartDict(final_chart,los)
        # else:
        self.create_Dict(final_meds, final_proc, final_out, final_chart, los)

    def create_chartDict(self, chart, los):
        dataDic = {}
        for hid in self.hids:
            grp = self.data[self.data["stay_id"] == hid]
            dataDic[hid] = {"Chart": {}, "label": int(grp["label"])}
        for hid in tqdm(self.hids):
            # CHART
            if self.feat_chart:
                df2 = chart[chart["stay_id"] == hid]
                val = df2.pivot_table(index="start_time", columns="itemid", values="valuenum")
                df2["val"] = 1
                df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
                # print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2 = pd.concat([df2, add_df])
                df2 = df2.sort_index()
                df2 = df2.fillna(0)

                val = pd.concat([val, add_df])
                val = val.sort_index()
                if self.impute == "Mean":
                    val = val.ffill()
                    val = val.bfill()
                    val = val.fillna(val.mean())
                elif self.impute == "Median":
                    val = val.ffill()
                    val = val.bfill()
                    val = val.fillna(val.median())
                val = val.fillna(0)

                df2[df2 > 0] = 1
                df2[df2 < 0] = 0
                # print(df2.head())
                dataDic[hid]["Chart"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")  # type: ignore [index]
                dataDic[hid]["Chart"]["val"] = val.iloc[:, 0:].to_dict(orient="list")  # type: ignore [index]

        # --- SAVE DICTIONARIES ---
        with open(os.path.join(self._out_dict_dir, "metaDic"), "rb") as fp:
            metaDic = pickle.load(fp)  # nosec: B301

        with open(os.path.join(self._out_dict_dir, "dataChartDic"), "wb") as fp:
            self.dataChartDic = dataDic
            pickle.dump(self.dataChartDic, fp)

        with open(os.path.join(self._out_dict_dir, "chartVocab"), "wb") as fp:
            self.chartVocab = list(chart["itemid"].unique())
            pickle.dump(self.chartVocab, fp)
        self.chart_vocab_n = chart["itemid"].nunique()
        metaDic["Chart"] = self.chart_per_adm

        with open(os.path.join(self._out_dict_dir, "metaDic"), "wb") as fp:
            self.metaDic = metaDic
            pickle.dump(self.metaDic, fp)

    def create_Dict(self, meds, proc, out, chart, los):
        dataDic = {}
        print(los)
        labels_csv = pd.DataFrame(columns=["stay_id", "label"])
        labels_csv["stay_id"] = pd.Series(self.hids)
        labels_csv["label"] = 0
        # print("# Unique gender",self.data.gender.nunique())
        # print("# Unique ethnicity",self.data.ethnicity.nunique())
        # print("# Unique insurance",self.data.insurance.nunique())

        for hid in self.hids:
            grp = self.data[self.data["stay_id"] == hid]
            dataDic[hid] = {
                "Cond": {},
                "Proc": {},
                "Med": {},
                "Out": {},
                "Chart": {},
                "ethnicity": grp["ethnicity"].iloc[0],
                "age": int(grp["Age"]),
                "gender": grp["gender"].iloc[0],
                "label": int(grp["label"]),
            }
            labels_csv.loc[labels_csv["stay_id"] == hid, "label"] = int(grp["label"])

            # print(static_csv.head())
        for hid in tqdm(self.hids):
            grp = self.data[self.data["stay_id"] == hid]
            demo_csv = grp[["Age", "gender", "ethnicity", "insurance"]]
            if not os.path.exists(os.path.join(self._out_csv_dir, str(hid))):
                os.makedirs(os.path.join(self._out_csv_dir, str(hid)))
            demo_csv.to_csv(os.path.join(self._out_csv_dir, str(hid), "demo.csv"), index=False)

            dyn_csv = pd.DataFrame()
            # MEDS
            if self.feat_med:
                feat = meds["itemid"].unique()
                df2 = meds[meds["stay_id"] == hid]
                if df2.shape[0] == 0:
                    amount = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                    amount = amount.fillna(0)
                    amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])
                else:
                    rate = df2.pivot_table(index="start_time", columns="itemid", values="rate")
                    # print(rate)
                    amount = df2.pivot_table(index="start_time", columns="itemid", values="amount")
                    df2 = df2.pivot_table(index="start_time", columns="itemid", values="stop_time")
                    # print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2 = pd.concat([df2, add_df])
                    df2 = df2.sort_index()
                    df2 = df2.ffill()
                    df2 = df2.fillna(0)

                    rate = pd.concat([rate, add_df])
                    rate = rate.sort_index()
                    rate = rate.ffill()
                    rate = rate.fillna(-1)

                    amount = pd.concat([amount, add_df])
                    amount = amount.sort_index()
                    amount = amount.ffill()
                    amount = amount.fillna(-1)
                    # print(df2.head())
                    df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                    df2[df2 > 0] = 1
                    df2[df2 < 0] = 0
                    rate.iloc[:, 0:] = df2.iloc[:, 0:] * rate.iloc[:, 0:]
                    amount.iloc[:, 0:] = df2.iloc[:, 0:] * amount.iloc[:, 0:]
                    # print(df2.head())
                    dataDic[hid]["Med"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
                    dataDic[hid]["Med"]["rate"] = rate.iloc[:, 0:].to_dict(orient="list")
                    dataDic[hid]["Med"]["amount"] = amount.iloc[:, 0:].to_dict(orient="list")

                    feat_df = pd.DataFrame(columns=list(set(feat) - set(amount.columns)))
                    # print(feat)
                    # print(amount.columns)
                    # print(amount.head())
                    amount = pd.concat([amount, feat_df], axis=1)

                    amount = amount[feat]
                    amount = amount.fillna(0)
                    # print(amount.columns)
                    amount.columns = pd.MultiIndex.from_product([["MEDS"], amount.columns])

                if dyn_csv.empty:
                    dyn_csv = amount
                else:
                    dyn_csv = pd.concat([dyn_csv, amount], axis=1)

            # PROCS
            if self.feat_proc:
                feat = proc["itemid"].unique()
                df2 = proc[proc["stay_id"] == hid]
                if df2.shape[0] == 0:
                    df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                    df2 = df2.fillna(0)
                    df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])
                else:
                    df2["val"] = 1
                    # print(df2)
                    df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
                    # print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2 = pd.concat([df2, add_df])
                    df2 = df2.sort_index()
                    df2 = df2.fillna(0)
                    df2[df2 > 0] = 1
                    # print(df2.head())
                    dataDic[hid]["Proc"] = df2.to_dict(orient="list")

                    feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                    df2 = pd.concat([df2, feat_df], axis=1)

                    df2 = df2[feat]
                    df2 = df2.fillna(0)
                    df2.columns = pd.MultiIndex.from_product([["PROC"], df2.columns])

                if dyn_csv.empty:
                    dyn_csv = df2
                else:
                    dyn_csv = pd.concat([dyn_csv, df2], axis=1)

            # OUT
            if self.feat_out:
                feat = out["itemid"].unique()
                df2 = out[out["stay_id"] == hid]
                if df2.shape[0] == 0:
                    df2 = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                    df2 = df2.fillna(0)
                    df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])
                else:
                    df2["val"] = 1
                    df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
                    # print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2 = pd.concat([df2, add_df])
                    df2 = df2.sort_index()
                    df2 = df2.fillna(0)
                    df2[df2 > 0] = 1
                    # print(df2.head())
                    dataDic[hid]["Out"] = df2.to_dict(orient="list")

                    feat_df = pd.DataFrame(columns=list(set(feat) - set(df2.columns)))
                    df2 = pd.concat([df2, feat_df], axis=1)

                    df2 = df2[feat]
                    df2 = df2.fillna(0)
                    df2.columns = pd.MultiIndex.from_product([["OUT"], df2.columns])

                if dyn_csv.empty:
                    dyn_csv = df2
                else:
                    dyn_csv = pd.concat([dyn_csv, df2], axis=1)

            # CHART
            if self.feat_chart:
                feat = chart["itemid"].unique()
                df2 = chart[chart["stay_id"] == hid]
                if df2.shape[0] == 0:
                    val = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                    val = val.fillna(0)
                    val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])
                else:
                    val = df2.pivot_table(index="start_time", columns="itemid", values="valuenum")
                    df2["val"] = 1
                    df2 = df2.pivot_table(index="start_time", columns="itemid", values="val")
                    # print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2 = pd.concat([df2, add_df])
                    df2 = df2.sort_index()
                    df2 = df2.fillna(0)

                    val = pd.concat([val, add_df])
                    val = val.sort_index()
                    if self.impute == "Mean":
                        val = val.ffill()
                        val = val.bfill()
                        val = val.fillna(val.mean())
                    elif self.impute == "Median":
                        val = val.ffill()
                        val = val.bfill()
                        val = val.fillna(val.median())
                    val = val.fillna(0)

                    df2[df2 > 0] = 1
                    df2[df2 < 0] = 0
                    # print(df2.head())
                    dataDic[hid]["Chart"]["signal"] = df2.iloc[:, 0:].to_dict(orient="list")
                    dataDic[hid]["Chart"]["val"] = val.iloc[:, 0:].to_dict(orient="list")

                    feat_df = pd.DataFrame(columns=list(set(feat) - set(val.columns)))
                    val = pd.concat([val, feat_df], axis=1)

                    val = val[feat]
                    val = val.fillna(0)
                    val.columns = pd.MultiIndex.from_product([["CHART"], val.columns])

                if dyn_csv.empty:
                    dyn_csv = val
                else:
                    dyn_csv = pd.concat([dyn_csv, val], axis=1)

            # Save temporal data to csv
            dyn_csv.to_csv(os.path.join(self._out_csv_dir, str(hid), "dynamic.csv"), index=False)

            # COND
            if self.feat_cond:
                feat = self.cond["new_icd_code"].unique()
                grp = self.cond[self.cond["stay_id"] == hid]
                if grp.shape[0] == 0:
                    dataDic[hid]["Cond"] = {"fids": list(["<PAD>"])}
                    feat_df = pd.DataFrame(np.zeros([1, len(feat)]), columns=feat)
                    grp = feat_df.fillna(0)
                    grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
                else:
                    dataDic[hid]["Cond"] = {"fids": list(grp["new_icd_code"])}
                    grp["val"] = 1
                    grp = grp.drop_duplicates()
                    grp = grp.pivot(index="stay_id", columns="new_icd_code", values="val").reset_index(drop=True)
                    feat_df = pd.DataFrame(columns=list(set(feat) - set(grp.columns)))
                    grp = pd.concat([grp, feat_df], axis=1)
                    grp = grp.fillna(0)
                    grp = grp[feat]
                    grp.columns = pd.MultiIndex.from_product([["COND"], grp.columns])
            grp.to_csv(os.path.join(self._out_csv_dir, str(hid), "static.csv"), index=False)
            labels_csv.to_csv(os.path.join(self._out_csv_dir, "labels.csv"), index=False)

        # --- SAVE DICTIONARIES ---
        metaDic = {"Cond": {}, "Proc": {}, "Med": {}, "Out": {}, "Chart": {}, "LOS": {}}  # type: ignore
        metaDic["LOS"] = los
        with open(os.path.join(self._out_dict_dir, "dataDic"), "wb") as fp:
            self.dataDic = dataDic
            pickle.dump(self.dataDic, fp)

        with open(os.path.join(self._out_dict_dir, "hadmDic"), "wb") as fp:
            self.hadmDic = self.hids
            pickle.dump(self.hadmDic, fp)

        with open(os.path.join(self._out_dict_dir, "ethVocab"), "wb") as fp:
            self.ethVocab = list(self.data["ethnicity"].unique())
            pickle.dump(self.ethVocab, fp)
            self.eth_vocab_n = self.data["ethnicity"].nunique()

        with open(os.path.join(self._out_dict_dir, "ageVocab"), "wb") as fp:
            self.ageVocab = list(self.data["Age"].unique())
            pickle.dump(self.ageVocab, fp)
            self.age_vocab_n = self.data["Age"].nunique()

        with open(os.path.join(self._out_dict_dir, "insVocab"), "wb") as fp:
            self.insVocab = list(self.data["insurance"].unique())
            pickle.dump(self.insVocab, fp)
            self.ins_vocab_n = self.data["insurance"].nunique()

        if self.feat_med:
            with open(os.path.join(self._out_dict_dir, "medVocab"), "wb") as fp:
                self.medVocab = list(meds["itemid"].unique())
                pickle.dump(self.medVocab, fp)
            self.med_vocab_n = meds["itemid"].nunique()
            metaDic["Med"] = self.med_per_adm

        if self.feat_out:
            with open(os.path.join(self._out_dict_dir, "outVocab"), "wb") as fp:
                self.outVocab = list(out["itemid"].unique())
                pickle.dump(self.outVocab, fp)
            self.out_vocab_n = out["itemid"].nunique()
            metaDic["Out"] = self.out_per_adm

        if self.feat_chart:
            with open(os.path.join(self._out_dict_dir, "chartVocab"), "wb") as fp:
                self.chartVocab = list(chart["itemid"].unique())
                pickle.dump(self.chartVocab, fp)
            self.chart_vocab_n = chart["itemid"].nunique()
            metaDic["Chart"] = self.chart_per_adm

        if self.feat_cond:
            with open(os.path.join(self._out_dict_dir, "condVocab"), "wb") as fp:
                self.condVocab = list(self.cond["new_icd_code"].unique())
                pickle.dump(self.condVocab, fp)
            self.cond_vocab_n = self.cond["new_icd_code"].nunique()
            metaDic["Cond"] = self.cond_per_adm  # type: ignore

        if self.feat_proc:
            with open(os.path.join(self._out_dict_dir, "procVocab"), "wb") as fp:
                self.procVocab = list(proc["itemid"].unique())
                pickle.dump(self.procVocab, fp)
            self.proc_vocab_n = proc["itemid"].nunique()
            metaDic["Proc"] = self.proc_per_adm

        with open(os.path.join(self._out_dict_dir, "metaDic"), "wb") as fp:
            self.metaDic = metaDic
            pickle.dump(self.metaDic, fp)
