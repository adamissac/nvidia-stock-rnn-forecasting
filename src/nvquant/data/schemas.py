"""pandera schemas for raw and aligned data. Validation failures stop the pipeline."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

_ascending = pa.Check(
    lambda idx: bool(idx.is_monotonic_increasing and idx.is_unique),
    element_wise=False,
    error="index must be strictly increasing (ascending time, no duplicates)",
)

OHLCV_SCHEMA = pa.DataFrameSchema(
    {
        "open": pa.Column(float, pa.Check.gt(0), nullable=True),
        "high": pa.Column(float, pa.Check.gt(0), nullable=True),
        "low": pa.Column(float, pa.Check.gt(0), nullable=True),
        "close": pa.Column(float, pa.Check.gt(0), nullable=True),
        "volume": pa.Column(float, pa.Check.ge(0), nullable=True),
    },
    index=pa.Index(pa.DateTime, checks=_ascending, name="date"),
    checks=[
        pa.Check(
            lambda df: bool(((df["high"] >= df["low"]) | df["high"].isna()).all()),
            element_wise=False,
            error="high < low",
        ),
    ],
    strict=True,
    coerce=True,
)

RATES_SCHEMA = pa.DataFrameSchema(
    {r"^DGS\d+$": pa.Column(float, pa.Check.in_range(-5, 25), nullable=True, regex=True)},
    index=pa.Index(pa.DateTime, checks=_ascending),
    coerce=True,
)

FACTORS_SCHEMA = pa.DataFrameSchema(
    {
        c: pa.Column(float, pa.Check.in_range(-0.5, 0.5), nullable=True)
        for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]
    },
    index=pa.Index(pa.DateTime, checks=_ascending),
    strict=True,
    coerce=True,
)


def validate_ohlcv(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Validate one OHLCV frame; raises ``pandera.errors.SchemaError`` with the ticker."""
    try:
        return OHLCV_SCHEMA.validate(frame)
    except pa.errors.SchemaError as exc:
        raise pa.errors.SchemaError(exc.schema, exc.data, f"{ticker}: {exc}") from exc
