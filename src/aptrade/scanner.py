"""Example live trading strategy using Massive API feeds.

This strategy monitors 10k+ stocks in real-time and trades based on
simple momentum signals.
"""

import datetime as dt
import threading
import time
import traceback
from zoneinfo import ZoneInfo

from aptrade.analyzers.eq import Eq
from aptrade.analyzers.tradeanalyzer import TradeAnalyzer
from aptrade.brokers.ibbroker import IBBroker
from aptrade.cerebro import Cerebro
from aptrade.core.alerts import AlertEvent, get_alert_publisher
from aptrade.core.config import settings
from aptrade.feeds.massive_live import setup_massive_live_feeds
from aptrade.plot import Plot
from aptrade.private.strategies.ORB import OpeningRangeBreakout
from aptrade.sizers import SimpleSizer

logger = get_alert_publisher()


def _scanner_telegram_format(event: AlertEvent) -> str:
    titles = {
        "scanner_window_entered": "Scanner Started",
        "scanner_window_exited": "Scanner Stopped",
        "scanner_runtime_restarted": "Scanner Restarted",
        "scanner_crash": "Scanner Error",
    }
    title = titles.get(event.name, event.name)
    return f"APTrade Alert\n{title}\n{event.message}"


def _is_within_window(
    now_local: dt.datetime,
    start_time: dt.time,
    stop_time: dt.time,
) -> bool:
    """Return True when the current local datetime is inside the scan window."""
    # Do not run scanner sessions on weekends.
    if now_local.weekday() >= 5:
        return False

    now = now_local.time()
    if start_time <= stop_time:
        return start_time <= now < stop_time
    # Supports windows that cross midnight.
    return now >= start_time or now < stop_time


def _build_scanner_runtime(
    api_key: str,
    max_symbols: int,
    min_volume: float,
    min_price: float | None,
    max_price: float | None,
):
    """Create a fresh Cerebro+feed runtime for one scanner session."""
    logger.info("[scanner-checkpoint] build_runtime:start")
    cerebro = Cerebro()

    storekwargs = {
        "host": "10.1.1.84",
        "port": 7497,
        # clientId=None,
        # account="None",
        "timeoffset": True,
        "timeout": 10.0,
        "_debug": False,
    }

    print("Using New IBstore")
    broker = IBBroker(**storekwargs)
    # broker = CustomBroker(**storekwargs)

    logger.info("[scanner-checkpoint] build_runtime:after_ibbroker_init")
    cerebro.setbroker(broker)

    logger.info(f"Setting up live feeds (max {max_symbols} symbols)...")
    manager = setup_massive_live_feeds(
        cerebro,
        api_key=api_key,
        update_interval=60,
        max_symbols=max_symbols,
        min_volume=min_volume,
        min_price=min_price,
        max_price=max_price,
        auto_start=True,
        # use_mock=True,
    )
    logger.info(
        f"[scanner] Polling running after setup: {manager.is_polling_running()}"
    )
    feed_count = manager.get_feed_count()
    logger.info(
        f"Created {feed_count} live feeds; cerebro currently has {len(cerebro.datas)} registered datas"
    )
    if feed_count == 0 or len(cerebro.datas) == 0:
        logger.warning(
            "Scanner runtime initialized without active feeds; strategy startup will likely be a no-op"
        )

    # logger.info("[scanner-checkpoint] build_runtime:before_ibbroker_init")

    logger.info("[scanner-checkpoint] build_runtime:after_setbroker")
    # cerebro.addstrategy(OpeningRangeBreakoutTest)
    cerebro.addstrategy(OpeningRangeBreakout)

    logger.info("[scanner-checkpoint] build_runtime:after_addstrategy")
    cerebro.addsizer(SimpleSizer, percents=10)
    logger.info("[scanner-checkpoint] build_runtime:after_addsizer")
    cerebro.addanalyzer(Eq, _name="eq")
    logger.info("[scanner-checkpoint] build_runtime:after_addanalyzer_eq")
    cerebro.addanalyzer(TradeAnalyzer, _name="trade_analyzer")
    logger.info("[scanner-checkpoint] build_runtime:after_addanalyzer_trade")
    logger.info(
        f"Scanner strategy configured with {len(cerebro.datas)} datas before cerebro.run()"
    )
    logger.info("[scanner-checkpoint] build_runtime:complete")

    # cerebro.broker.setcash(2000)
    return cerebro, manager


def _shutdown_scanner_runtime(cerebro, manager, runner_thread, reason: str) -> None:
    """Gracefully stop feeds and strategy execution for the active session."""
    logger.info(f"Stopping scanner runtime: {reason}")

    manager.stop_polling()
    manager.stop_feeds()
    cerebro.runstop()

    if runner_thread and runner_thread.is_alive():
        logger.info("Waiting for strategy execution to complete...")
        runner_thread.join(timeout=30)
        if runner_thread.is_alive():
            logger.warning("Thread still alive after timeout")
        else:
            logger.info("Strategy execution completed")

    today_str = dt.datetime.now().strftime("%Y%m%d")
    portfolio_value = cerebro.broker.getvalue()
    try:
        portfolio_value_text = f"{float(portfolio_value):.2f}"
    except (TypeError, ValueError):
        portfolio_value_text = str(portfolio_value)
    logger.info(f"Final Portfolio Value: {portfolio_value_text}")

    try:
        flat_runstrats = [
            strat for stratlist in cerebro.runstrats for strat in stratlist
        ]
        Plot().plot(
            flat_runstrats,
            filename=f"/home/vcaldas/trading/aptrade/results/smacrossbt_{today_str}.html",
            show=False,
        )
        logger.info(
            f"Saved chart to /home/vcaldas/trading/aptrade/results/smacrossbt_{today_str}.html"
        )
    except Exception as e:
        logger.error(f"Failed to generate plot: {e}", exc_info=True)

    try:
        cerebro.show_report(
            filename=f"/home/vcaldas/trading/aptrade/results/smacrossbt_stats_{today_str}.html",
            show=False,
        )
        logger.info(
            f"Saved report to /home/vcaldas/trading/aptrade/results/smacrossbt_stats_{today_str}.html"
        )
    except Exception as e:
        logger.error(f"Failed to generate report: {e}", exc_info=True)

    stats = manager.get_stats()
    logger.info("=" * 80)
    logger.info("FINAL STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Feeds created: {stats['feed_count']}")
    logger.info(f"Total updates: {stats['update_count']}")
    logger.info(f"Last update: {stats['last_update_time']}")
    logger.info(f"Symbols updated in last cycle: {stats['symbols_updated']}")
    logger.info(f"Symbols skipped in last cycle: {stats['symbols_skipped']}")
    logger.info("=" * 80)


def run_live_scanner(
    api_key: str,
    max_symbols: int = 10,
    min_volume: float = 1_000_000,
    min_price: float | None = 1.0,
    max_price: float | None = 100.0,
    start_time: dt.time = dt.time(4, 0),  # 4 AM ET
    stop_time: dt.time = dt.time(20, 0),  # 8 PM ET
    scanner_timezone: str = settings.SCANNER_TIMEZONE,
):
    """Run the live scanner strategy based on time window.

    Args:
        api_key: Massive.io API key
        max_symbols: Maximum number of symbols to track
        min_volume: Minimum previous day volume
        min_price: Minimum stock price
        max_price: Maximum stock price
        start_time: Time to start scanning (ET timezone)
        stop_time: Time to stop scanning (ET timezone)
    """
    # Explicit timezone keeps scheduling deterministic across servers.
    scan_tz = ZoneInfo(scanner_timezone)

    # Run based on time window
    logger.info(
        name="scanner_startup",
        message=f"Starting live scanner: {start_time} - {stop_time} ({scanner_timezone})",
        notify_telegram=True,
        telegram_formatter=_scanner_telegram_format,
    )
    logger.info("Press Ctrl+C to stop early")

    runner_thread = None
    running = False
    cerebro = None
    manager = None
    runtime_error: Exception | None = None
    runtime_error_traceback: str | None = None
    was_in_window = False
    waiting_for_feeds_logged = False

    try:

        def run_cerebro(active_cerebro):
            nonlocal runtime_error, runtime_error_traceback
            try:
                logger.info("[scanner-checkpoint] run_cerebro:before_cerebro_run")
                # For live trading with continuous polling, run indefinitely
                # This keeps the strategy alive to process new bars as they arrive
                active_cerebro.run()
                logger.info("[scanner-checkpoint] run_cerebro:after_cerebro_run")
            except Exception as e:
                runtime_error = e
                runtime_error_traceback = traceback.format_exc()
                logger.error(
                    name="scanner_crash",
                    message=f"Scanner runtime crashed: {e}",
                    notify_telegram=True,
                    telegram_formatter=_scanner_telegram_format,
                    exc_info=True,
                )
            except KeyboardInterrupt:
                logger.info("Interrupted by user")

        def clear_runtime() -> None:
            nonlocal cerebro, manager, runner_thread, running
            runner_thread = None
            cerebro = None
            manager = None
            running = False

        def shutdown_runtime(reason: str) -> None:
            nonlocal cerebro, manager, runner_thread, running
            if cerebro and manager:
                _shutdown_scanner_runtime(
                    cerebro,
                    manager,
                    runner_thread,
                    reason=reason,
                )
            clear_runtime()

        # Normal operation: use time window
        while True:
            now_local = dt.datetime.now(scan_tz)
            now_time = now_local.time()
            in_window = _is_within_window(now_local, start_time, stop_time)

            if in_window and not was_in_window:
                logger.info(
                    name="scanner_window_entered",
                    message=f"Scanner started for trading window at {now_time} ({scanner_timezone})",
                    notify_telegram=True,
                    telegram_formatter=_scanner_telegram_format,
                )
                cerebro, manager = _build_scanner_runtime(
                    api_key=api_key,
                    max_symbols=max_symbols,
                    min_volume=min_volume,
                    min_price=min_price,
                    max_price=max_price,
                )
                runtime_error = None
                runtime_error_traceback = None
                waiting_for_feeds_logged = False

            if in_window and not running and cerebro and manager:
                # Start polling if not already running
                if not manager.is_polling_running():
                    logger.warning(
                        "Polling thread is not running; restarting polling to restore feed updates"
                    )
                    manager.start_polling()
                else:
                    logger.info("[scanner] Polling thread is already running")

                feed_count = manager.get_feed_count()
                if feed_count > 0:
                    runner_thread = threading.Thread(
                        target=run_cerebro,
                        args=(cerebro,),
                        daemon=False,
                    )
                    runner_thread.start()
                    running = True
                    waiting_for_feeds_logged = False
                    logger.info(
                        f"[scanner] Starting scanner runtime with {feed_count} active feeds, polling_running={manager.is_polling_running()}"
                    )
                elif not waiting_for_feeds_logged:
                    logger.info(
                        "No feeds available yet; waiting for polling updates before starting scanner runtime"
                    )
                    waiting_for_feeds_logged = True

            if not in_window and was_in_window:
                logger.info(
                    name="scanner_window_exited",
                    message=f"Scanner stopped because trading window closed at {now_time} ({scanner_timezone})",
                    notify_telegram=True,
                    telegram_formatter=_scanner_telegram_format,
                )
                shutdown_runtime(reason="trading window closed")
                waiting_for_feeds_logged = False

            if in_window and running and runner_thread and not runner_thread.is_alive():
                if runtime_error is not None:
                    logger.error(
                        "Scanner runtime exited during the active trading window due to %s: %s",
                        type(runtime_error).__name__,
                        runtime_error,
                    )
                    if runtime_error_traceback:
                        logger.error(
                            "Scanner runtime traceback:\n%s",
                            runtime_error_traceback,
                        )
                    shutdown_runtime(
                        reason=(
                            "runtime exception "
                            f"{type(runtime_error).__name__}: {runtime_error}"
                        )
                    )
                else:
                    logger.warning(
                        "Scanner runtime exited during the active trading window without an exception; it will remain stopped until the next window opens"
                    )
                    shutdown_runtime(
                        reason="unexpected runtime exit (no exception captured)"
                    )
                waiting_for_feeds_logged = False

            was_in_window = in_window

            if not in_window:
                waiting_for_feeds_logged = False
                logger.info(
                    f"Waiting for trading window ({now_time} is before {start_time} or after {stop_time})"
                )
                time.sleep(60)
                continue

            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        if cerebro and manager:
            _shutdown_scanner_runtime(
                cerebro,
                manager,
                runner_thread,
                reason="application shutdown",
            )


if __name__ == "__main__":
    # Get API key from environment or use hardcoded
    api_key = getattr(settings, "MASSIVE_API_KEY", None)
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is not configured")
    API_KEY = str(api_key)

    # Run live scanner
    run_live_scanner(
        api_key=API_KEY,
        max_symbols=10,
        min_volume=200_000,
        min_price=1.0,
        max_price=100,
        start_time=dt.time(4, 0),  # 4 AM ET
        stop_time=dt.time(16, 30),  # 4:30 PM ET
        scanner_timezone=settings.SCANNER_TIMEZONE,
    )
