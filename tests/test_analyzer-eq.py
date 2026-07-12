import datetime as dt

import numpy as np
import pandas as pd

from aptrade.analyzers.eq import (
    _safe_rms,
    _safe_sample_std,
    _timedelta_days,
    compute_drawdown_duration_peaks,
)


def test_timedelta_days_returns_nan_for_nan_values():
    assert np.isnan(_timedelta_days(np.nan))
    assert np.isnan(_timedelta_days(pd.NaT))


def test_timedelta_days_supports_timedelta_values():
    assert _timedelta_days(pd.Timedelta(days=2, hours=12)) == 2.5
    assert _timedelta_days(np.timedelta64(36, "h")) == 1.5
    assert _timedelta_days(dt.timedelta(hours=12)) == 0.5


def test_compute_drawdown_duration_peaks_handles_no_drawdown():
    index = pd.date_range("2026-01-01", periods=4, freq="D")
    dd = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)

    durations, peaks = compute_drawdown_duration_peaks(dd)

    assert durations.isna().all()
    assert peaks.isna().all()
    assert np.isnan(_timedelta_days(durations.max()))
    assert np.isnan(_timedelta_days(durations.mean()))


def test_safe_sample_std_returns_nan_for_short_inputs():
    assert np.isnan(_safe_sample_std([]))
    assert np.isnan(_safe_sample_std([1.0]))
    assert _safe_sample_std([1.0, 3.0]) == np.std([1.0, 3.0], ddof=1)


def test_safe_rms_returns_nan_for_empty_inputs():
    assert np.isnan(_safe_rms([]))
    assert _safe_rms([3.0, 4.0]) == np.sqrt(np.mean(np.square([3.0, 4.0])))
