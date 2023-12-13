from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from tempor.datasources.mivdp.preproc.cohort import day_intervals_cohort_v1
from tempor.datasources.mivdp.preproc.features import feature_preproc_icu

COMMON_CASES = [
    # No ICD code or disease label:
    dict(use_ICU=True, time=0, label="Mortality", icd_code=None, disease_label=None),
    dict(use_ICU=True, time=30, label="Readmission", icd_code=None, disease_label=None),
    dict(use_ICU=True, time=120, label="Readmission", icd_code=None, disease_label=None),
    dict(use_ICU=True, time=3, label="Length of Stay", icd_code=None, disease_label=None),
    dict(use_ICU=True, time=7, label="Length of Stay", icd_code=None, disease_label=None),
    # With ICD code:
    dict(use_ICU=True, time=0, label="Mortality", icd_code="I50", disease_label=None),
    dict(use_ICU=True, time=0, label="Mortality", icd_code="J44", disease_label=None),
    # With disease label:
    dict(use_ICU=True, time=0, label="Readmission", icd_code=None, disease_label="I50"),
    dict(use_ICU=True, time=0, label="Readmission", icd_code=None, disease_label="J44"),
]

EXPECTED_COLUMNS__FEATURE_ICU = {
    "diag": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "icd_code",
        "root_icd10_convert",
        "root",
    ],
    "out": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "intime",
        "event_time_from_admit",
    ],
    "chart": [
        "stay_id",
        "itemid",
        "event_time_from_admit",
        "valuenum",
    ],
    "proc": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "starttime",
        "intime",
        "event_time_from_admit",
    ],
    "med": [
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
    ],
}


class TestFeatureIcu:
    @pytest.mark.filterwarnings("ignore::pandas.errors.DtypeWarning")
    @pytest.mark.parametrize("args", COMMON_CASES)
    def test_common_cases(self, mimiciv_1_0_tiny: Tuple[Path, str], args):
        root_dir, version = mimiciv_1_0_tiny

        # NOTE: This should really be taken out of here in order to be more unit test -like, and to make tests faster.
        cohort, cohort_output = day_intervals_cohort_v1.extract_data(  # pylint: disable=unused-variable
            version=version,
            root_dir=root_dir.as_posix(),
            cohort_output=None,
            summary_output=None,
            **args,
        )

        # This is the actual test:
        dfs = feature_preproc_icu.feature_icu(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            version=version,
            diag_flag=True,
            out_flag=True,
            chart_flag=True,
            proc_flag=True,
            med_flag=True,
        )

        dfs_names = list(EXPECTED_COLUMNS__FEATURE_ICU.keys())

        output_dir = root_dir / "data" / "features"
        assert output_dir.exists()

        for df_name in dfs_names:
            assert (output_dir / f"preproc_{df_name}_icu.csv.gz").exists()

        for name, df in zip(dfs_names, dfs):
            assert isinstance(df, pd.DataFrame)

            assert len(df) > 0
            assert len(df.columns) > 0

            # The returned dataframes here are a superset (in terms of columns) compared to what is actually
            # saved into files. The EXPECTED_... contains only the columns that are saved into files.
            # Hence the set comparison.
            assert set(df.columns.tolist()) >= set(EXPECTED_COLUMNS__FEATURE_ICU[name])


EXPECTED_COLUMNS__PREPROCESS_FEATURES_ICU__GROUPING = {
    "both": ["subject_id", "hadm_id", "stay_id", "new_icd_code"],
    "convert": ["subject_id", "hadm_id", "stay_id", "new_icd_code"],
    "convert_group": ["subject_id", "hadm_id", "stay_id", "new_icd_code"],
}


class TestPreprocessFeaturesIcu:
    @pytest.mark.parametrize("group_diag", ["both", "convert", "convert_group"])
    @pytest.mark.parametrize("args", COMMON_CASES)
    def test_grouping_common_cases(self, mimiciv_1_0_tiny: Tuple[Path, str], args, group_diag):
        root_dir, version = mimiciv_1_0_tiny

        # NOTE: This should really be taken out of here in order to be more unit test -like, and to make tests faster.
        cohort, cohort_output = day_intervals_cohort_v1.extract_data(  # pylint: disable=unused-variable
            version=version,
            root_dir=root_dir.as_posix(),
            cohort_output=None,
            summary_output=None,
            **args,
        )
        dfs = feature_preproc_icu.feature_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            version=version,
            diag_flag=True,
            out_flag=False,  # Not needed for this test.
            chart_flag=False,  # (!) Not needed for this test.
            proc_flag=False,  # Not needed for this test.
            med_flag=False,  # Not needed for this test.
        )

        # This is the actual test:
        diag, chart = feature_preproc_icu.preprocess_features_icu(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            group_diag=group_diag,
            chart_flag=False,
            clean_chart=False,
            impute_outlier_chart=False,
            thresh=0,
            left_thresh=0,
        )

        assert chart is None

        assert isinstance(diag, pd.DataFrame)

        assert len(diag) > 0
        assert len(diag.columns) > 0

        print(diag.columns.tolist())
        assert sorted(diag.columns.tolist()) == sorted(EXPECTED_COLUMNS__PREPROCESS_FEATURES_ICU__GROUPING[group_diag])

    @pytest.mark.filterwarnings("ignore::pandas.errors.DtypeWarning")
    @pytest.mark.filterwarnings("ignore::pandas.errors.SettingWithCopyWarning")
    @pytest.mark.parametrize("args", COMMON_CASES)
    def test_cleaning_common_cases(self, mimiciv_1_0_tiny: Tuple[Path, str], args):
        root_dir, version = mimiciv_1_0_tiny

        # NOTE: This should really be taken out of here in order to be more unit test -like, and to make tests faster.
        cohort, cohort_output = day_intervals_cohort_v1.extract_data(  # pylint: disable=unused-variable
            version=version,
            root_dir=root_dir.as_posix(),
            cohort_output=None,
            summary_output=None,
            **args,
        )
        dfs = feature_preproc_icu.feature_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            version=version,
            diag_flag=False,  # (!) Not needed for this test.
            out_flag=False,  # Not needed for this test.
            chart_flag=True,
            proc_flag=False,  # Not needed for this test.
            med_flag=False,  # Not needed for this test.
        )

        # This is the actual test:
        diag, chart = feature_preproc_icu.preprocess_features_icu(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=False,
            group_diag="both",  # N/A for this test.
            chart_flag=True,
            clean_chart=True,
            impute_outlier_chart=True,
            thresh=5,
            left_thresh=95,
        )

        assert diag is None

        assert isinstance(chart, pd.DataFrame)

        assert len(chart) > 0
        assert len(chart.columns) > 0

        print(chart.columns.tolist())
        assert sorted(chart.columns.tolist()) == sorted(EXPECTED_COLUMNS__FEATURE_ICU["chart"])


EXPECTED_COLUMNS__GENERATE_SUMMARY_ICU = {
    "diag": [
        "new_icd_code",
        "mean_frequency",
        "total_count",
    ],
    "out": [
        "itemid",
        "mean_frequency",
        "missing_count",
        "total_count",
    ],
    "chart": [
        "itemid",
        "mean_frequency",
        "total_count",
    ],
    "proc": [
        "itemid",
        "mean_frequency",
        "total_count",
    ],
    "med": [
        "itemid",
        "mean_frequency",
        "missing_count",
        "total_count",
    ],
}


class TestGenerateSummaryIcu:
    @pytest.mark.filterwarnings("ignore::pandas.errors.DtypeWarning")
    @pytest.mark.parametrize("args", COMMON_CASES)
    def test_common_cases(self, mimiciv_1_0_tiny: Tuple[Path, str], args):
        root_dir, version = mimiciv_1_0_tiny

        # NOTE: This should really be taken out of here in order to be more unit test -like, and to make tests faster.
        cohort, cohort_output = day_intervals_cohort_v1.extract_data(  # pylint: disable=unused-variable
            version=version,
            root_dir=root_dir.as_posix(),
            cohort_output=None,
            summary_output=None,
            **args,
        )
        dfs = feature_preproc_icu.feature_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            version=version,
            diag_flag=True,
            out_flag=True,
            chart_flag=True,
            proc_flag=True,
            med_flag=True,
        )
        diag, chart = feature_preproc_icu.preprocess_features_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            group_diag="both",
            chart_flag=False,
            clean_chart=False,
            impute_outlier_chart=False,
            thresh=0,
            left_thresh=0,
        )

        # This is the actual test:
        summary_dfs = feature_preproc_icu.generate_summary_icu(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            out_flag=True,
            chart_flag=True,
            proc_flag=True,
            med_flag=True,
        )

        dfs_names = list(EXPECTED_COLUMNS__GENERATE_SUMMARY_ICU.keys())

        output_dir = root_dir / "data" / "summary"
        assert output_dir.exists()

        for df_name in dfs_names:
            assert (output_dir / f"{df_name}_summary.csv").exists()
            assert (output_dir / f"{df_name}_features.csv").exists()

        for name, df in zip(dfs_names, summary_dfs):
            assert isinstance(df, pd.DataFrame)

            assert len(df) > 0
            assert len(df.columns) > 0

            assert sorted(df.columns.tolist()) == sorted(EXPECTED_COLUMNS__GENERATE_SUMMARY_ICU[name])


EXPECTED_COLUMNS__FEATURES_SELECTION_ICU = {
    "diag": ["hadm_id", "new_icd_code", "stay_id", "subject_id"],
    "out": EXPECTED_COLUMNS__FEATURE_ICU["out"],
    "chart": EXPECTED_COLUMNS__FEATURE_ICU["chart"],
    "proc": EXPECTED_COLUMNS__FEATURE_ICU["proc"],
    "med": EXPECTED_COLUMNS__FEATURE_ICU["med"],
}


class TestFeaturesSelectionIcu:
    @pytest.mark.filterwarnings("ignore::pandas.errors.DtypeWarning")
    @pytest.mark.parametrize("args", COMMON_CASES)
    def test_common_cases(self, mimiciv_1_0_tiny: Tuple[Path, str], args):
        root_dir, version = mimiciv_1_0_tiny

        # NOTE: This should really be taken out of here in order to be more unit test -like, and to make tests faster.
        cohort, cohort_output = day_intervals_cohort_v1.extract_data(  # pylint: disable=unused-variable
            version=version,
            root_dir=root_dir.as_posix(),
            cohort_output=None,
            summary_output=None,
            **args,
        )
        dfs = feature_preproc_icu.feature_icu(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            version=version,
            diag_flag=True,
            out_flag=True,
            chart_flag=True,
            proc_flag=True,
            med_flag=True,
        )
        diag, chart = feature_preproc_icu.preprocess_features_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            group_diag="both",
            chart_flag=False,
            clean_chart=False,
            impute_outlier_chart=False,
            thresh=0,
            left_thresh=0,
        )
        summary_dfs = feature_preproc_icu.generate_summary_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            out_flag=True,
            chart_flag=True,
            proc_flag=True,
            med_flag=True,
        )

        # This is the actual test:
        dfs = feature_preproc_icu.features_selection_icu(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            proc_flag=True,
            med_flag=True,
            out_flag=True,
            chart_flag=True,
            select_diag=True,
            select_proc=True,
            select_med=True,
            select_out=True,
            select_chart=True,
        )

        dfs_names = list(EXPECTED_COLUMNS__FEATURES_SELECTION_ICU.keys())

        output_dir = root_dir / "data" / "features"
        assert output_dir.exists()

        for df_name in dfs_names:
            assert (output_dir / f"preproc_{df_name}_icu.csv.gz").exists()

        for name, df in zip(dfs_names, dfs):
            assert isinstance(df, pd.DataFrame)

            assert len(df) > 0
            assert len(df.columns) > 0

            assert sorted(df.columns.tolist()) == sorted(EXPECTED_COLUMNS__FEATURES_SELECTION_ICU[name])
