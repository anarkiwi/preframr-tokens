"""Unit test: tighten_persist_dtypes round-trips correctly + actually
narrows the column dtypes.
"""

import os
import tempfile

import pandas as pd

from preframr_tokens.utils import tighten_persist_dtypes
from preframr_tokens.stfconstants import DESCRIPTION_PDTYPE, SUBREG_PDTYPE


def _df_with_int64_macro_cols(n=100):
    rows = []
    for i in range(n):
        rows.append(
            {
                "reg": -128,
                "val": i,
                "subreg": -1 if i % 3 else (i % 5),
                "description": 0,
                "diff": 32,
                "irq": 19656,
                "op": 0,
            }
        )
    df = pd.DataFrame(rows)
    df["op"] = df["op"].astype("UInt8")
    df["reg"] = df["reg"].astype("Int8")
    df["val"] = df["val"].astype("Int32")
    df["diff"] = df["diff"].astype("UInt16")
    df["irq"] = df["irq"].astype("UInt16")
    return df


def test_tighten_changes_dtypes():
    df = _df_with_int64_macro_cols()
    assert df["subreg"].dtype == "int64"
    assert df["description"].dtype == "int64"
    tighten_persist_dtypes(df)
    assert df["subreg"].dtype == SUBREG_PDTYPE
    assert df["description"].dtype == DESCRIPTION_PDTYPE


def test_tighten_preserves_values():
    df = _df_with_int64_macro_cols()
    orig_subreg = df["subreg"].tolist()
    orig_desc = df["description"].tolist()
    tighten_persist_dtypes(df)
    assert df["subreg"].tolist() == orig_subreg
    assert df["description"].tolist() == orig_desc


def test_tighten_idempotent():
    df = _df_with_int64_macro_cols()
    tighten_persist_dtypes(df)
    sub_after_first = df["subreg"].dtype
    tighten_persist_dtypes(df)
    assert df["subreg"].dtype == sub_after_first


def test_tighten_roundtrip_through_parquet():
    df = _df_with_int64_macro_cols()
    tighten_persist_dtypes(df)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.parquet")
        df.to_parquet(p, engine="pyarrow", compression="zstd")
        back = pd.read_parquet(p)
        assert back["subreg"].dtype == SUBREG_PDTYPE
        assert back["description"].dtype == DESCRIPTION_PDTYPE
        assert back["subreg"].tolist() == df["subreg"].tolist()


def test_tighten_no_op_when_columns_missing():
    """Tightener should accept a df without subreg / description (e.g. a
    raw dump.parquet pre-macro-pipeline)."""
    df = pd.DataFrame({"reg": [0, 1, 2], "val": [10, 20, 30]})
    out = tighten_persist_dtypes(df)
    assert "subreg" not in out.columns
    assert "description" not in out.columns
