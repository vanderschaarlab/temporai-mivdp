import os
import shutil
from pathlib import Path
from typing import Tuple

import pytest

LOCAL_MIMIC_TEST_DATA_DIRS = {
    "1.0-tiny": os.path.realpath(os.path.join(os.path.dirname(__file__), "../data_root/mimiciv/1.0-tiny/")),
}


def get_data_version_for_tests(tmp_path_factory, version: str) -> Tuple[Path, str]:
    root_dir = tmp_path_factory.mktemp("data_root") / "mimiciv"
    test_dir = root_dir / version
    shutil.copytree(LOCAL_MIMIC_TEST_DATA_DIRS[version], test_dir)
    return root_dir, version


@pytest.fixture(scope="session")
def mimiciv_1_0_tiny(tmp_path_factory) -> Tuple[Path, str]:
    return get_data_version_for_tests(tmp_path_factory, "1.0-tiny")
