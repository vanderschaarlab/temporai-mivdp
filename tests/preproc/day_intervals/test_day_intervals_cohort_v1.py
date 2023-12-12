from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest

from tempor.datasources.mivdp.preproc.day_intervals import day_intervals_cohort_v1

EXPECTED_COLUMNS = [
    "Age",
    "dod",
    "ethnicity",
    "gender",
    "hadm_id",
    "insurance",
    "intime",
    "los",
    "min_valid_year",
    "outtime",
    "stay_id",
    "subject_id",
    "label",
]


class TestExtractData:
    @pytest.mark.parametrize(
        "args",
        [
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
        ],
    )
    def test_common_cases(self, mimiciv_1_0_tiny: Tuple[Path, str], args):
        root_dir, version = mimiciv_1_0_tiny

        cohort, cohort_output = day_intervals_cohort_v1.extract_data(
            version=version,
            root_dir=root_dir.as_posix(),
            cohort_output=None,
            summary_output=None,
            **args,
        )

        assert isinstance(cohort, pd.DataFrame)
        assert isinstance(cohort_output, str)

        assert len(cohort) > 0
        assert sorted(cohort.columns.tolist()) == sorted(EXPECTED_COLUMNS)

        assert args["label"].replace(" ", "_").lower() in cohort_output.lower()
        assert "icu" in cohort_output.lower()

        output_dir = root_dir / "data" / "cohort"
        assert output_dir.exists()
        assert (output_dir / f"{cohort_output}.csv.gz").exists()
        assert (output_dir / f"{cohort_output.replace('cohort', 'summary')}.txt").exists()
