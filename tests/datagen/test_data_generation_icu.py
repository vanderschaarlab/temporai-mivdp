from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from packaging.version import Version

from tempor.datasources.mivdp.datagen import data_generation_icu
from tempor.datasources.mivdp.preproc.cohort import day_intervals_cohort_v1
from tempor.datasources.mivdp.preproc.features import feature_preproc_icu

# Pytest utilities:

pandas_2 = Version(pd.__version__) >= Version("2.0.0")
conditional_filterwarnings = (
    pytest.mark.filterwarnings("ignore::pandas.errors.SettingWithCopyWarning") if pandas_2 else lambda x: x
)

# -------------------


COMMON_CASES = [
    # No ICD code or disease label:
    dict(use_ICU=True, time=0, label="Mortality", icd_code=None, disease_label=None),
    dict(use_ICU=True, time=30, label="Readmission", icd_code=None, disease_label=None),
    dict(use_ICU=True, time=3, label="Length of Stay", icd_code=None, disease_label=None),
    # With ICD code:
    dict(use_ICU=True, time=0, label="Mortality", icd_code="I50", disease_label=None),
    # With disease label:
    dict(use_ICU=True, time=0, label="Readmission", icd_code=None, disease_label="J44"),
]


class TestFeaturesSelectionIcu:
    @pytest.mark.filterwarnings("ignore:.*Calling.*int.*:FutureWarning")
    @pytest.mark.filterwarnings("ignore::pandas.errors.DtypeWarning")
    @conditional_filterwarnings
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
        summary_dfs = feature_preproc_icu.generate_summary_icu(  # pylint: disable=unused-variable  # noqa: F841
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            diag_flag=True,
            out_flag=True,
            chart_flag=True,
            proc_flag=True,
            med_flag=True,
        )
        dfs = feature_preproc_icu.features_selection_icu(  # noqa: F841
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

        # This is the actual test:
        data_gen = data_generation_icu.ICUDataGenerator(
            cohort_output=cohort_output,
            root_dir=root_dir.as_posix(),
            if_mort=args["label"] == "Mortality",
            if_admn=args["label"] == "Readmission",
            if_los=args["label"] == "Length of Stay",
            feat_cond=True,
            feat_proc=True,
            feat_out=True,
            feat_chart=True,
            feat_med=True,
            # TODO: Vary the below in test.
            impute="Mean",
            include_time=24,
            bucket=1,
            predW=6,
            silence_warnings=False,
        )

        assert isinstance(data_gen.data, pd.DataFrame)
        assert isinstance(data_gen.cond, pd.DataFrame)
        assert isinstance(data_gen.proc, pd.DataFrame)
        assert isinstance(data_gen.out, pd.DataFrame)
        assert isinstance(data_gen.chart, pd.DataFrame)
        assert isinstance(data_gen.meds, pd.DataFrame)
        assert len(data_gen.data) > 0
        assert len(data_gen.cond) > 0
        assert len(data_gen.proc) > 0
        assert len(data_gen.out) > 0
        assert len(data_gen.chart) > 0
        assert len(data_gen.meds) > 0
        # TODO: More asserts.
