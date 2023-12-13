"""Common utilities."""

from typing import Any

import pandas as pd
from packaging.version import Version


# NOTE: Inefficient approach, but this was used in the original code. The whole approach should be redone.
def pd_v2_compat_append(df_append_to: pd.DataFrame, new_item: Any, **kwargs) -> pd.DataFrame:
    """Append to dataframe, with compatibility for pandas versions < 2.0.0.

    Args:
        df_append_to (pd.DataFrame): Dataframe to append to.
        new_item (Any): New content being appended.

    Returns:
        pd.DataFrame: Dataframe with appended data.
    """
    if Version(pd.__version__) < Version("2.0.0"):
        return df_append_to.append(new_item, **kwargs)  # type: ignore
    else:
        to_append = pd.DataFrame([new_item]) if not isinstance(new_item, pd.DataFrame) else new_item
        return pd.concat([df_append_to, to_append], **kwargs)
