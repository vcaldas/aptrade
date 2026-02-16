"""Example live trading strategy using Massive API feeds.

This strategy monitors 10k+ stocks in real-time and trades based on
simple momentum signals.
"""

import datetime as dt
import logging
import time
from typing import Optional
from zoneinfo import ZoneInfo

import aptrade as bt
from aptrade.analyzers.tradeanalyzer import TradeAnalyzer
from aptrade.feeds.massive_live import setup_massive_live_feeds
from aptrade.sizer import SimpleSizer
from aptrade.strategies.ORB import OpeningRangeBreakout

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def run_live_scanner(
    api_key: str,
    max_symbols: int = 100,
    min_volume: float = 1_000_000,
    min_price: Optional[float] = 5.0,
    max_price: Optional[float] = 100.0,
    start_time: dt.time = dt.time(4, 0),  # 4 AM ET
    stop_time: dt.time = dt.time(20, 0),  # 8 PM ET
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
    logger.info("=" * 80)
    logger.info("LIVE MARKET SCANNER - Massive API")
    logger.info("=" * 80)

    # Eastern timezone
    et_tz = ZoneInfo("America/New_York")

    # Create cerebro
    cerebro = bt.Cerebro()

    # Set up live feeds
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
    )
    logger.info(f"Created {manager.get_feed_count()} live feeds")

    # Add strategy
    cerebro.addstrategy(OpeningRangeBreakout)
    cerebro.addsizer(SimpleSizer, percents=10)
    cerebro.addanalyzer(TradeAnalyzer, _name="trade_analyzer")

    # Set initial cash
    cerebro.broker.setcash(2000)

    # Run based on time window
    logger.info(f"Starting live scanner: {start_time} - {stop_time} ET")
    logger.info("Press Ctrl+C to stop early")

    runner_thread = None
    running = False

    try:
        import threading

        def run_cerebro():
            try:
                cerebro.run()
            except Exception as e:
                logger.error(f"Exception in run_cerebro: {e}", exc_info=True)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")

        # Normal operation: use time window
        while True:
            now_et = dt.datetime.now(et_tz).time()

            # Check if we're within the trading window
            if start_time <= now_et < stop_time:
                if not running:
                    logger.info(f"Entering trading window at {now_et}")
                    runner_thread = threading.Thread(
                        target=run_cerebro,
                        daemon=False,  # Not daemon - we want to wait
                    )
                    runner_thread.start()
                    running = True

                # Sleep briefly before checking again
                time.sleep(5)
            else:
                if running:
                    logger.info(f"Exiting trading window at {now_et}")
                    # Request stop and wait for cerebro to finish
                    cerebro.runstop()
                    if runner_thread and runner_thread.is_alive():
                        logger.info("Waiting for cerebro to finish...")
                        runner_thread.join(timeout=30)
                        logger.info("Cerebro finished")
                    running = False

                    # Sleep longer when outside trading window
                    logger.info(
                        f"Waiting for trading window ({now_et} is before {start_time} or after {stop_time})"
                    )
                    time.sleep(60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        logger.info("Stopping polling...")
        manager.stop_polling()

        # Signal all feeds to stop
        logger.info("Signaling all feeds to stop...")
        manager.stop_feeds()

        # Signal cerebro to stop running
        logger.info("Signaling cerebro to stop...")
        cerebro.runstop()

        # Wait for runner thread to complete if it's still running
        if runner_thread and runner_thread.is_alive():
            logger.info("Waiting for strategy execution to complete...")
            runner_thread.join(timeout=30)
            if runner_thread.is_alive():
                logger.warning("Thread still alive after timeout")
            else:
                logger.info("Strategy execution completed")

        # get today date for filename
        today_str = dt.datetime.now().strftime("%Y%m%d")
        logger.info(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

        # Generate plots and reports
        try:
            cerebro.plot(
                filename=f"/home/vcaldas/aptrade/results/smacrossbt_{today_str}.html",
                show=False,
            )
            logger.info(
                f"Saved chart to /home/vcaldas/aptrade/results/smacrossbt_{today_str}.html"
            )
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}", exc_info=True)

        try:
            cerebro.show_report(
                filename=f"/home/vcaldas/aptrade/results/smacrossbt_stats_{today_str}.html",
                show=False,
            )
            logger.info(
                f"Saved report to /home/vcaldas/aptrade/results/smacrossbt_stats_{today_str}.html"
            )
        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)
        # Print stats
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


if __name__ == "__main__":
    from aptrade.core.config import settings

    # Get API key from environment or use hardcoded
    api_key = getattr(settings, "MASSIVE_API_KEY", None)
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is not configured")
    API_KEY = str(api_key)

    # Run live scanner
    run_live_scanner(
        api_key=API_KEY,
        max_symbols=30,
        min_volume=500_000,
        min_price=None,
        max_price=50.0,
        start_time=dt.time(4, 0),  # 4 AM ET
        stop_time=dt.time(16, 30),  # 4:30 PM ET
    )
