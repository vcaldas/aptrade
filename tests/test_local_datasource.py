import pandas as pd
import pytest

from aptrade.datasource import local


def _write_parquet(dir_path, filename, rows):
    dir_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(dir_path / filename)


def test_get_minute_ohlcv_happy_path(tmp_path, monkeypatch):
    base = tmp_path / "data"
    monkeypatch.setattr(local, "DATA_DIR", base)

    trades_dir = base / "interim" / "us_stocks_sip" / "A" / "trades"
    ts_1430_05 = pd.Timestamp("2025-01-02T14:30:05Z").value
    ts_1430_20 = pd.Timestamp("2025-01-02T14:30:20Z").value
    ts_1431_05 = pd.Timestamp("2025-01-02T14:31:05Z").value
    ts_1431_55 = pd.Timestamp("2025-01-02T14:31:55Z").value

    _write_parquet(
        trades_dir,
        "20250102.trades.0000.parquet",
        [
            {
                "participant_timestamp": ts_1430_05,
                "price": 100.0,
                "size": 50,
                "tape": 3,
            },
            {
                "participant_timestamp": ts_1430_20,
                "price": 101.0,
                "size": 100,
                "tape": 3,
            },
            {
                "participant_timestamp": ts_1431_05,
                "price": 102.0,
                "size": 200,
                "tape": 3,
            },
            {
                "participant_timestamp": ts_1431_55,
                "price": 103.0,
                "size": 100,
                "tape": 3,
            },
        ],
    )

    result = local.get_minute_ohlcv("A", "20250102")

    expected = [
        {
            "time": int(pd.Timestamp("2025-01-02T14:30:00Z").value // 1_000_000),
            "open": 100.0,
            "high": 101.0,
            "low": 100.0,
            "close": 101.0,
            "volume": 150.0,
        },
        {
            "time": int(pd.Timestamp("2025-01-02T14:31:00Z").value // 1_000_000),
            "open": 102.0,
            "high": 103.0,
            "low": 102.0,
            "close": 103.0,
            "volume": 300.0,
        },
    ]

    assert result == expected


def test_get_minute_ohlcv_missing_files(tmp_path, monkeypatch):
    base = tmp_path / "data"
    monkeypatch.setattr(local, "DATA_DIR", base)

    with pytest.raises(FileNotFoundError):
        local.get_minute_ohlcv("A", "20250102")


def test_get_minute_ohlcv_missing_columns(tmp_path, monkeypatch):
    base = tmp_path / "data"
    monkeypatch.setattr(local, "DATA_DIR", base)

    trades_dir = base / "interim" / "us_stocks_sip" / "A" / "trades"
    ts_1430_05 = pd.Timestamp("2025-01-02T14:30:05Z").value

    # Missing 'size' column should raise KeyError during resampling
    _write_parquet(
        trades_dir,
        "20250102.trades.0000.parquet",
        [
            {"participant_timestamp": ts_1430_05, "price": 100.0, "tape": 3},
        ],
    )

    with pytest.raises(KeyError):
        local.get_minute_ohlcv("A", "20250102")
