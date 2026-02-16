import datetime
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple, Type

import aptrade as bt
from aptrade.strategies.ORB import OpeningRangeBreakout

__all__ = ["run_backtest"]

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_TEMPLATE = "data/stocks/{ticker}.trades.parquet"
_DEFAULT_EXPORT_DIR = "backtest_trades"
_DEFAULT_DATA_PARAMS: Dict[str, Any] = {
    "datetime": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "openinterest": -1,
    "timeframe": "Minutes",
    "tzinput": "UTC",
    "tz": "America/New_York",
}


def run_backtest(config: Mapping[str, Any], debug: bool = False) -> Dict[str, float]:
    """Run a batch of backtests based on the given configuration.

    Parameters
    ----------
    config:
        Mapping with optional keys:

        - ``tickers`` / ``symbols``: sequence or comma-separated list of assets.
        - ``tickers_file``: file whose non-empty lines provide tickers.
        - ``assets_dir``: directory containing files whose names (or stems) are tickers.
        - ``data_template`` (or ``data_path_template``/``data_path``):
          format string with ``{ticker}`` placeholder pointing to the Parquet feed.
        - ``data_root`` / ``base_dir``: base path applied when templates are relative.
        - ``export_dir``: directory for analyzer exports (default ``backtest_trades``).
        - ``export_root``: base directory used when ``export_dir`` is relative.
        - ``start_cash`` / ``cash``: starting capital (default ``10000``).
        - ``stake`` / ``position_size``: fixed stake per order (default ``100``).
        - ``commission``: commission rate (default ``0.0005``).
        - ``strategy``: dotted path to a ``bt.Strategy`` subclass (default ``OpeningRangeBreakout``).
        - ``strategy_params``: mapping with strategy keyword arguments.
        - ``data``: mapping overriding ``GenericParquetData`` keyword arguments.
        - ``cerebro``: mapping forwarded to ``cerebro.run``.
        - ``analyzers``: iterable of mappings with ``path`` (import path), optional
          ``name`` and ``params`` keys to add extra analyzers.
        - ``skip_trade_history``: disable attaching ``TradeHistoryAnalyzer`` when true.
        - ``show_progress``: log per-ticker progress (default ``True``).
        - ``log_level``: override logger level (string or logging level integer).

    debug:
        When ``True`` keep running after individual ticker failures and log stack traces.

    Returns
    -------
    Dict[str, float]
        Mapping of ticker symbol to total return (fractional, e.g. ``0.12`` for 12%).
    """

    cfg: MutableMapping[str, Any] = dict(config)
    _initialise_logger(cfg.get("log_level"), debug)

    tickers = _resolve_tickers(cfg)
    if not tickers:
        logger.warning("No tickers provided; skipping backtest run.")
        return {}

    results: Dict[str, float] = {}
    total = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
        try:
            if cfg.get("show_progress", True):
                logger.info("Processing %s (%s/%s)", ticker, index, total)
            results[ticker] = _run_single_ticker(ticker, cfg)
        except Exception as exc:  # pragma: no cover - defensive logging path
            if debug:
                logger.exception("Backtest failed for %s: %s", ticker, exc)
                continue
            raise

    return results


def _run_single_ticker(ticker: str, config: Mapping[str, Any]) -> float:
    cerebro = bt.Cerebro()

    strategy_cls, strategy_kwargs = _resolve_strategy(config)
    cerebro.addstrategy(strategy_cls, **strategy_kwargs)

    data_kwargs = dict(_DEFAULT_DATA_PARAMS)
    data_overrides = config.get("data") or {}
    if not isinstance(data_overrides, Mapping):
        raise TypeError("'data' configuration must be a mapping.")
    data_kwargs.update(data_overrides)

    timeframe = data_kwargs.pop("timeframe", "Minutes")
    data_kwargs["timeframe"] = _coerce_timeframe(timeframe)
    if "compression" in data_kwargs:
        data_kwargs["compression"] = int(data_kwargs["compression"])

    data_path = _resolve_data_path(ticker, config)
    data_kwargs["dataname"] = str(data_path)

    data_feed = bt.feeds.GenericParquetData(**data_kwargs)
    cerebro.adddata(data_feed, name=ticker)

    starting_cash = float(config.get("cash", config.get("start_cash", 10000)))
    cerebro.broker.setcash(starting_cash)
    stake_value = int(config.get("stake", config.get("position_size", 100)))
    cerebro.addsizer(bt.sizer.FixedSize, stake=stake_value)
    cerebro.broker.setcommission(commission=float(config.get("commission", 0.0005)))

    if not config.get("skip_trade_history"):
        export_dir = _resolve_path(
            config.get("export_dir", _DEFAULT_EXPORT_DIR),
            base=config.get("export_root")
            or config.get("base_dir")
            or config.get("data_root"),
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        # print(export_dir)
        # cerebro.addanalyzer(
        #     TradeHistoryAnalyzer,
        #     _name="trade_history",
        #     export_dir=str(export_dir),
        # )

    for analyzer_conf in config.get("analyzers", []):
        if not isinstance(analyzer_conf, Mapping):
            raise TypeError("Each analyzer configuration must be a mapping.")
        path = analyzer_conf.get("path")
        if not path:
            raise ValueError("Analyzer configuration requires a 'path' key.")
        analyzer_cls = _import_from_string(str(path))
        analyzer_name = analyzer_conf.get("name")
        analyzer_params = analyzer_conf.get("params") or {}
        if not isinstance(analyzer_params, Mapping):
            raise TypeError("Analyzer 'params' must be a mapping when provided.")
        if analyzer_name:
            cerebro.addanalyzer(
                analyzer_cls, _name=str(analyzer_name), **analyzer_params
            )
        else:
            cerebro.addanalyzer(analyzer_cls, **analyzer_params)
    cerebro.dtcerebro = datetime.datetime.now()
    cerebro_kwargs = config.get("cerebro") or {}
    if not isinstance(cerebro_kwargs, Mapping):
        raise TypeError("'cerebro' configuration must be a mapping.")
    cerebro.run(**dict(cerebro_kwargs))

    final_value = cerebro.broker.getvalue()
    if starting_cash == 0:
        return 0.0
    return (final_value - starting_cash) / starting_cash


def _resolve_tickers(config: Mapping[str, Any]) -> Sequence[str]:
    explicit = config.get("tickers") or config.get("symbols")
    tickers = _coerce_ticker_sequence(explicit)
    if tickers:
        return tickers

    tickers_file = config.get("tickers_file")
    if tickers_file:
        file_path = _resolve_path(tickers_file, base=config.get("base_dir"))
        if not file_path.exists():
            raise FileNotFoundError(f"Tickers file '{file_path}' not found.")
        tickers = [
            line.strip() for line in file_path.read_text().splitlines() if line.strip()
        ]
        return _dedupe_preserve_order(tickers)

    assets_dir = config.get("assets_dir")
    if assets_dir:
        directory = _resolve_path(
            assets_dir, base=config.get("assets_root") or config.get("base_dir")
        )
        if not directory.exists():
            raise FileNotFoundError(f"Assets directory '{directory}' not found.")
        entries = sorted(directory.iterdir())
        tickers = [entry.stem if entry.is_file() else entry.name for entry in entries]
        return _dedupe_preserve_order(tickers)

    return []


def _resolve_strategy(
    config: Mapping[str, Any],
) -> Tuple[Type[bt.Strategy], Dict[str, Any]]:
    strategy_spec = config.get("strategy")
    if strategy_spec:
        strategy_cls = _import_from_string(str(strategy_spec))
    else:
        strategy_cls = OpeningRangeBreakout

    strategy_params = config.get("strategy_params") or {}
    if not isinstance(strategy_params, Mapping):
        raise TypeError("'strategy_params' configuration must be a mapping.")

    return strategy_cls, dict(strategy_params)


def _resolve_data_path(ticker: str, config: Mapping[str, Any]) -> Path:
    template = (
        config.get("data_template")
        or config.get("data_path_template")
        or config.get("data_path")
        or _DEFAULT_DATA_TEMPLATE
    )
    try:
        formatted = str(template).format(ticker=ticker)
    except KeyError as exc:
        raise KeyError(
            f"Data template '{template}' is missing placeholder: {exc}."
        ) from exc

    data_root = config.get("data_root") or config.get("base_dir")
    data_path = _resolve_path(formatted, base=data_root)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file '{data_path}' for ticker '{ticker}' not found."
        )
    return data_path


def _coerce_timeframe(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return getattr(bt.TimeFrame, value)
        except AttributeError as exc:
            raise ValueError(f"Unknown timeframe '{value}'.") from exc
    raise TypeError("Timeframe must be an int or name of bt.TimeFrame attribute.")


def _coerce_ticker_sequence(value: Any) -> Sequence[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set, frozenset)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return _dedupe_preserve_order(items)

    if isinstance(value, str):
        items = [part.strip() for part in value.split(",") if part.strip()]
        return _dedupe_preserve_order(items)

    if isinstance(value, Iterable):
        items = [str(item).strip() for item in value if str(item).strip()]
        return _dedupe_preserve_order(items)

    raise TypeError("Tickers must be provided as a sequence or comma-separated string.")


def _dedupe_preserve_order(items: Sequence[str]) -> Sequence[str]:
    seen = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _resolve_path(path_value: Any, base: Any | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    base_path = Path(base).expanduser() if base else _PROJECT_ROOT
    return base_path / path


def _import_from_string(dotted_path: str):
    module_path, _, attribute = dotted_path.replace(":", ".").rpartition(".")
    if not module_path:
        raise ValueError(f"Unable to import '{dotted_path}': missing module path.")
    module = import_module(module_path)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - narrow failure path
        raise ImportError(
            f"Module '{module_path}' has no attribute '{attribute}'."
        ) from exc


def _initialise_logger(level: Any, debug: bool) -> None:
    desired = level
    if desired is None:
        desired = logging.DEBUG if debug else logging.INFO
    elif isinstance(desired, str):
        desired = getattr(logging, desired.upper(), logging.INFO)
    elif not isinstance(desired, int):
        desired = logging.INFO

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(int(desired))
