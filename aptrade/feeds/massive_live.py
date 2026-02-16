"""Live data feed from Massive.io API for 10k+ stocks.

This module provides a backtrader-compatible data feed that polls the Massive.io
API every minute to get real-time snapshots of all stocks.
"""

import logging
import random
import time
from datetime import datetime, timezone
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Dict, List, Optional

import pytz
from massive import RESTClient
from massive.rest.models import (
    Agg,
    TickerSnapshot,
)

import aptrade as bt

logger = logging.getLogger(__name__)


class MockMassiveAPI:
    """Mock Massive API for testing when markets are closed.

    Generates realistic-looking ticker snapshots with random price movements.
    Useful for testing and development on weekends/after hours.
    """

    def __init__(self, num_tickers: int = 100, base_price: float = 100.0):
        """Initialize mock API.

        Args:
            num_tickers: Number of tickers to generate
            base_price: Base price for ticker generation
        """
        self.num_tickers = num_tickers
        self.base_price = base_price
        self._tickers = [f"MOCK{i:04d}" for i in range(num_tickers)]
        self._prices = {}
        self._initialize_prices()

    def _initialize_prices(self):
        """Initialize starting prices for all tickers."""
        import random

        for ticker in self._tickers:
            self._prices[ticker] = self.base_price * (0.5 + random.random() * 1.5)

    def _create_agg(self, price: float, volatility: float = 0.02):
        """Create a mock Agg object with realistic OHLC."""
        import random

        class Agg:
            def __init__(self):
                variation = price * volatility
                self.open = price + random.uniform(-variation, variation)
                self.high = max(self.open, price) * (1 + random.uniform(0, volatility))
                self.low = min(self.open, price) * (1 - random.uniform(0, volatility))
                self.c = self.close = self.open + random.uniform(-variation, variation)
                # Ensure OHLC constraints
                self.high = max(self.high, self.open, self.close)
                self.low = min(self.low, self.open, self.close)
                self.volume = random.randint(100_000, 10_000_000)

        return Agg()

    def get_snapshot_all(self, asset_class: str = "stocks"):
        """Mock get_snapshot_all endpoint.

        Args:
            asset_class: Asset class (ignored in mock)

        Returns:
            List of mock TickerSnapshot objects
        """

        class TickerSnapshot:
            def __init__(self, ticker, prev_day, min_bar, updated):
                self.ticker = ticker
                self.prev_day = prev_day
                self.min = min_bar
                self.updated = updated

        snapshots = []
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        for ticker in self._tickers:
            # Update price with random walk
            current_price = self._prices[ticker]
            price_change = random.uniform(-0.03, 0.03)  # +/- 3% per update
            new_price = current_price * (1 + price_change)
            self._prices[ticker] = max(new_price, 0.01)  # Don't go below 1 cent

            # Create prev_day and minute bar
            prev_day = self._create_agg(current_price, volatility=0.05)
            min_bar = self._create_agg(self._prices[ticker], volatility=0.01)

            snapshot = TickerSnapshot(
                ticker=ticker, prev_day=prev_day, min_bar=min_bar, updated=now_ms
            )
            snapshots.append(snapshot)

        logger.debug(f"Mock API returned {len(snapshots)} snapshots")
        return snapshots


class MassiveAPIClient:
    """Wrapper for Massive.io API to fetch stock snapshots.

    Handles API calls, error recovery, and rate limiting for the get_snapshot_all
    endpoint which returns ~10k stocks with OHLCV data.

    Can also use a mock API for testing when markets are closed.
    """

    def __init__(
        self,
        api_key: str,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        use_mock: bool = False,
        mock_tickers: int = 100,
    ):
        """Initialize the Massive API client.

        Args:
            api_key: Massive.io API key (ignored if use_mock=True)
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Seconds to wait between retries
            use_mock: Use mock API instead of real Massive API
            mock_tickers: Number of mock tickers to generate
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_mock = use_mock

        if use_mock:
            logger.info(f"Using MOCK Massive API with {mock_tickers} tickers")
            self._client = MockMassiveAPI(num_tickers=mock_tickers)
        else:
            self._client: Optional[RESTClient] = None

        self._last_fetch_time: Optional[datetime] = None
        self._last_snapshot_count: int = 0

    def _get_client(self) -> RESTClient:
        """Lazy initialization of REST client."""
        if self.use_mock:
            return self._client
        if self._client is None:
            self._client = RESTClient(self.api_key)
        return self._client

    def fetch_snapshots(self) -> List[TickerSnapshot]:
        """Fetch all stock snapshots from Massive API.

        Returns:
            List of TickerSnapshot objects with current market data

        Raises:
            Exception: If all retry attempts fail
        """
        client = self._get_client()

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                # snapshots = client.get_snapshot_all("stocks")
                snapshots = client.get_snapshot_direction("stocks", direction="gainers")
                elapsed = time.time() - start_time

                for i, item in enumerate(snapshots):
                    # verify this is an TickerSnapshot
                    if isinstance(item, TickerSnapshot):
                        # verify this is an Agg
                        if isinstance(item.prev_day, Agg):
                            # verify this is a float
                            if isinstance(
                                item.prev_day.open, float | int
                            ) and isinstance(item.prev_day.close, float | int):
                                percent_change = (  # noqa F841
                                    (item.day.high - item.prev_day.close)
                                    / item.prev_day.close
                                    * 100
                                )
                                # print(item)
                                # print(
                                #     "{:<15}{:<15}{:<15}{:.2f}{:.2f} %".format(
                                #         item.ticker,
                                #         item.day.high,
                                #         item.prev_day.close,
                                #         percent_change,
                                #         item.todays_change_percent,
                                #     )
                                # )

                # Filter for valid snapshots (use duck typing - must have ticker attribute)
                valid_snapshots = [
                    snap for snap in snapshots if hasattr(snap, "ticker")
                ]

                self._last_fetch_time = datetime.now(timezone.utc)
                self._last_snapshot_count = len(valid_snapshots)

                logger.info(
                    f"Fetched {len(valid_snapshots)} stock snapshots "
                    f"in {elapsed:.2f}s"
                )
                return valid_snapshots

            except Exception as e:
                logger.error(
                    f"API fetch attempt {attempt + 1}/{self.max_retries} failed: {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.critical(
                        f"All API fetch attempts failed after {self.max_retries} tries"
                    )
                    raise

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics.

        Returns:
            Dict with last_fetch_time and snapshot_count
        """
        return {
            "last_fetch_time": self._last_fetch_time,
            "snapshot_count": self._last_snapshot_count,
        }

    def get_aggregates(
        self,
        ticker: str,
        # multiplier: int,
        # timespan: str,
        # start_date: str,
        # end_date: str,
        adjusted: str = "true",
        sort: str = "asc",
        limit: int = 1200,
    ) -> List[Agg]:
        # Get date range (e.g., last 25 days)
        end_date = datetime.now()
        client = self._get_client()

        aggs = []
        for a in client.list_aggs(
            ticker,
            1,
            "minute",
            end_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            adjusted="true",
            sort="asc",
            limit=1200,
        ):
            aggs.append(a)
        return aggs


def convert_timestamp_to_et(timestamp_ms):
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    dt_et = dt.astimezone(pytz.timezone("America/New_York"))
    return dt_et


def extract_ohlcv(snapshot: TickerSnapshot) -> Optional[Agg]:
    """Extract OHLCV data from a TickerSnapshot as an Agg.

    Args:
        snapshot: TickerSnapshot from Massive API (or compatible object with ticker, prev_day, min attributes)

    Returns:
        Agg with open, high, low, close, volume, timestamp set
        Returns None if data is invalid or incomplete
    """
    # Check if we have required fields (duck typing for testing)
    if not hasattr(snapshot, "ticker") or not getattr(snapshot, "ticker", None):
        return None

    # Extract current minute bar if available
    minute_bar = getattr(snapshot, "min", None)
    prev_day = getattr(snapshot, "prev_day", None)

    # We need at least prev_day data for initial price (duck typing - check for required attributes)
    if prev_day is None or not hasattr(prev_day, "close"):
        return None

    # Use minute bar if available, otherwise use prev_day (duck typing)
    if minute_bar is not None and hasattr(minute_bar, "close"):
        bar = minute_bar
    else:
        print(f"{snapshot.ticker}Using prev_day data for OHLCV")
        # Fallback to prev_day if no minute data
        bar = prev_day

    # ALWAYS use current time instead of API timestamp
    # The API's 'updated' field often contains stale timestamps
    dt = datetime.now(timezone.utc)

    # Log if API timestamp is very old for debugging
    if hasattr(snapshot, "updated") and isinstance(snapshot.updated, (int, float)):
        try:
            api_dt = datetime.fromtimestamp(snapshot.updated / 1000, tz=timezone.utc)
            age = (dt - api_dt).total_seconds()
            if age > 300:  # More than 5 minutes old
                logger.debug(
                    f"{snapshot.ticker}: API timestamp is {age:.0f}s old, using current time instead"
                )
        except (TypeError, ValueError, OSError):
            pass

    # # Ensure required OHLCV fields are present
    # if getattr(bar, "close", None) is None:
    #     return None

    # if getattr(bar, "open", None) is None:
    #     bar.open = bar.close
    # if getattr(bar, "high", None) is None:
    #     bar.high = bar.close
    # if getattr(bar, "low", None) is None:
    #     bar.low = bar.close
    # if getattr(bar, "volume", None) is None:
    #     bar.volume = 0.0

    # Only set timestamp if it doesn't exist (for snapshots without proper timestamps)
    # Historical aggregates already have correct timestamps - preserve them
    # Always set timestamp to current time for live snapshot data
    # This ensures the feed shows the actual time the data was fetched, not stale API timestamps
    bar.timestamp = int(dt.timestamp() * 1000)
    return bar


def validate_ohlcv(bar: Agg) -> bool:
    """Validate OHLCV data integrity.

    Args:
        bar: Agg with OHLCV fields

    Returns:
        True if data passes validation, False otherwise
    """
    try:
        o, h, l, c, v = (  # noqa: E741
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )

        if o is None or h is None or l is None or c is None or v is None:
            logger.debug("Invalid OHLCV: missing values")
            return False

        o = float(o)
        h = float(h)
        l = float(l)  # noqa: E741
        c = float(c)
        v = float(v)

        # Basic OHLCV constraints
        if h < l:
            logger.warning("Invalid OHLCV: high < low")
            return False

        if h < max(o, c) or l > min(o, c):
            logger.warning("Invalid OHLCV: high/low doesn't contain open/close")
            return False

        if v < 0:
            logger.warning("Invalid OHLCV: negative volume")
            return False

        # Check for reasonable prices (not zero or negative)
        if any(x <= 0 for x in [o, h, l, c]):
            logger.debug("Invalid OHLCV: zero or negative prices")
            return False

        return True

    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Validation error: {e}")
        return False


class MassiveLiveData(bt.DataBase):
    """Backtrader data feed for a single symbol from Massive API.

    This feed is designed to work with the MassiveFeedManager which polls the API
    every minute and pushes new data to all symbol feeds.
    """

    params = (
        ("ticker", None),  # Stock ticker symbol
        ("prev_day_close", None),  # Previous day close for initial value
    )

    def __init__(self):
        """Initialize the data feed for a single symbol."""
        super(MassiveLiveData, self).__init__()

        # Thread-safe data buffer using queue for historical + live data
        self._bar_queue: Queue[Agg] = Queue()
        self._has_new_data = Event()
        self._should_stop = Event()  # Signal to stop feeding data

        # Note: prev_day_close is stored in params but not pushed as initial bar
        # Historical bars or live data will provide actual bars with correct timestamps

    def push_bar(self, bar: Agg):
        """Push new bar data to this feed (called by FeedManager).

        Args:
            bar: Agg with open, high, low, close, volume, timestamp
        """
        self._bar_queue.put(bar)
        self._has_new_data.set()

    def stop(self):
        """Signal the feed to stop loading data."""
        self._should_stop.set()

    def _load(self):
        """Load the next bar into backtrader's lines.

        For live feeds, this waits for new data from the manager's push_bar() method.

        Returns:
            True if new data was loaded, False otherwise
        """
        # Check if we should stop
        if self._should_stop.is_set():
            logger.debug(f"{self.p.ticker}: Stop signal received, stopping feed")
            return False

        # Check if we have queued data
        if self._bar_queue.empty():
            # Wait for new data with shorter timeout so we can respond to stop signals
            timeout = 5.0
            if not self._has_new_data.wait(timeout=timeout):
                logger.debug(f"{self.p.ticker}: Waiting for new data...")
                return None

            # Check again after wait
            if self._bar_queue.empty():
                return None

        # Get next bar from queue
        try:
            bar = self._bar_queue.get_nowait()
        except Exception as e:
            logger.warning(f"Error getting bar from queue: {e}")
            return None

        # Clear event only when queue is empty
        if self._bar_queue.empty():
            self._has_new_data.clear()

        # Convert timestamp from UTC to ET (US market timezone)
        ts = bar.timestamp
        dt_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        dt_et = dt_utc.astimezone(pytz.timezone("America/New_York"))

        # Convert to timezone-naive datetime for backtrader storage
        # We've already converted to ET, so just remove timezone info
        dt_naive = dt_et.replace(tzinfo=None)

        # Set backtrader's data lines with timezone-naive ET datetime
        self.lines.datetime[0] = bt.date2num(dt_naive)
        self.lines.open[0] = bar.open
        self.lines.high[0] = bar.high
        self.lines.low[0] = bar.low
        self.lines.close[0] = bar.close
        self.lines.volume[0] = bar.volume
        self.lines.openinterest[0] = 0

        logger.debug(f"{self.p.ticker}: Loaded bar at {dt_et} (${bar.close:.2f})")

        return True

    def islive(self):
        """Indicate this is a live data feed."""
        return True


class MassiveFeedManager:
    """Manages 10k+ live data feeds with 1-minute polling from Massive API.

    This manager:
    1. Polls get_snapshot_all() every 60 seconds
    2. Distributes data to individual symbol feeds
    3. Handles thread safety and error recovery
    """

    def __init__(
        self,
        api_key: str,
        update_interval: int = 60,
        max_symbols: Optional[int] = None,
        symbol_filter: Optional[callable] = None,
        use_mock: bool = False,
        mock_tickers: int = 100,
    ):
        """Initialize the feed manager.

        Args:
            api_key: Massive.io API key (ignored if use_mock=True)
            update_interval: Seconds between API polls (default 60)
            max_symbols: Maximum number of symbols to track (None = all)
            symbol_filter: Optional function to filter symbols (receives TickerSnapshot)
            use_mock: Use mock API for testing
            mock_tickers: Number of mock tickers to generate
        """
        self.api_client = MassiveAPIClient(
            api_key, use_mock=use_mock, mock_tickers=mock_tickers
        )
        self.update_interval = update_interval
        self.max_symbols = max_symbols
        self.symbol_filter = symbol_filter

        # Symbol -> DataFeed mapping
        self._feeds: Dict[str, MassiveLiveData] = {}
        self._feeds_lock = Lock()

        # Polling thread
        self._polling_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._running = False

        # Cerebro reference (set when initial feeds are created)
        self._cerebro: Optional[bt.Cerebro] = None

        # Stats
        self._update_count = 0
        self._last_update_time: Optional[datetime] = None
        self._symbols_updated = 0
        self._symbols_skipped = 0

    def create_feeds(self, cerebro: bt.Cerebro) -> int:
        """Fetch initial snapshot and create data feeds for all symbols.

        This should be called once before cerebro.run() to initialize all feeds.

        Args:
            cerebro: Backtrader Cerebro instance to add feeds to

        Returns:
            Number of feeds created
        """
        logger.info("Fetching initial snapshot to create feeds...")

        self._cerebro = cerebro

        try:
            snapshots = self.api_client.fetch_snapshots()
        except Exception as e:
            logger.error(f"Failed to fetch initial snapshot: {e}")
            return 0

        feeds_created = 0

        for snapshot in snapshots:
            # Apply symbol filter if provided
            if self.symbol_filter and not self.symbol_filter(snapshot):
                print("Filtered", snapshot.ticker)
                continue
            ticker = snapshot.ticker
            if not ticker:
                continue

            # Extract OHLCV data (optional for initial feed creation)
            bar = extract_ohlcv(snapshot)

            # Validate data - but create feed even if invalid (may get valid data later)
            is_valid = validate_ohlcv(bar) if bar else False
            # print(f"Ticker: {ticker}, is_valid: {is_valid}")
            # Create feed for this symbol
            # Use prev_day_close if available, otherwise use close (even if invalid)
            prev_day = getattr(snapshot, "prev_day", None)
            prev_day_close = getattr(prev_day, "close", None)
            if prev_day_close is None or prev_day_close <= 0:
                prev_day_close = None  # Let feed start without initial bar

            feed = MassiveLiveData(ticker=ticker, prev_day_close=prev_day_close)

            # Add to cerebro
            cerebro.adddata(feed, name=ticker)

            # Store in mapping
            with self._feeds_lock:
                self._feeds[ticker] = feed

            # Push existing historical bars so the strategy can process them
            aggs = self.api_client.get_aggregates(ticker)
            # print(f"Fetched {len(aggs)} aggregates for {ticker}")
            if aggs:
                for bar in aggs:
                    if validate_ohlcv(bar):
                        feed.push_bar(bar)
            elif not is_valid:
                logger.debug(
                    f"Created feed for {ticker} with no initial data (waiting for valid update)"
                )

            feeds_created += 1

            # Check max_symbols limit
            if self.max_symbols and feeds_created >= self.max_symbols:
                logger.info(f"Reached max_symbols limit of {self.max_symbols}")
                break

        logger.info(f"Created {feeds_created} live data feeds")
        return feeds_created

    def _add_feed_from_snapshot(self, snapshot, bar: Optional[Agg] = None) -> bool:
        """Create and register a feed for a new snapshot symbol.

        Returns True if a feed was created, False otherwise.
        """
        if self._cerebro is None:
            return False

        # Apply symbol filter if provided
        if self.symbol_filter and not self.symbol_filter(snapshot):
            return False

        ticker = getattr(snapshot, "ticker", None)
        if not ticker:
            return False

        with self._feeds_lock:
            if ticker in self._feeds:
                return False

        if self.max_symbols and self.get_feed_count() >= self.max_symbols:
            return False

        # Use prev_day_close if available, otherwise start without initial bar
        prev_day = getattr(snapshot, "prev_day", None)
        prev_day_close = getattr(prev_day, "close", None)
        if prev_day_close is None or prev_day_close <= 0:
            prev_day_close = None

        feed = MassiveLiveData(ticker=ticker, prev_day_close=prev_day_close)
        self._cerebro.adddata(feed, name=ticker)

        with self._feeds_lock:
            self._feeds[ticker] = feed

        # Seed with historical aggregates
        aggs = self.api_client.get_aggregates(ticker)
        if aggs:
            for agg in aggs:
                if validate_ohlcv(agg):
                    feed.push_bar(agg)

        # Push the current bar if valid
        if bar is not None and validate_ohlcv(bar):
            feed.push_bar(bar)

        logger.info(f"Added new live feed for {ticker}")
        return True

    def start_polling(self):
        """Start the background polling thread."""
        if self._running:
            logger.warning("Polling already running")
            return

        self._stop_event.clear()
        self._running = True
        self._polling_thread = Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()

        logger.info(f"Started polling thread (interval: {self.update_interval}s)")

    def stop_feeds(self):
        """Signal all feeds to stop loading data."""
        with self._feeds_lock:
            for feed in self._feeds.values():
                feed.stop()

    def get_feed_count(self) -> int:
        """Return the number of feeds managed."""
        with self._feeds_lock:
            return len(self._feeds)

    def stop_polling(self):
        """Stop the background polling thread."""
        if not self._running:
            return

        logger.info("Stopping polling thread...")
        self._stop_event.set()
        self._running = False

        if self._polling_thread:
            self._polling_thread.join(timeout=10)

        logger.info("Polling stopped")

    def _polling_loop(self):
        """Background thread that polls API every update_interval seconds."""
        while not self._stop_event.is_set():
            try:
                self._fetch_and_update()
            except Exception as e:
                logger.error(f"Error in polling loop: {e}", exc_info=True)

            # Wait for next interval (with early exit check)
            self._stop_event.wait(self.update_interval)

    def _fetch_and_update(self):
        """Fetch latest snapshots and update all feeds."""
        update_start = time.time()

        try:
            snapshots = self.api_client.fetch_snapshots()
        except Exception as e:
            logger.error(f"Failed to fetch snapshots: {e}")
            return

        updated = 0
        skipped = 0

        for snapshot in snapshots:
            # Extract data
            bar = extract_ohlcv(snapshot)
            ticker = snapshot.ticker
            if not ticker:
                skipped += 1
                continue

            # If feed doesn't exist yet, attempt to add it (even if bar is missing/invalid)
            with self._feeds_lock:
                feed = self._feeds.get(ticker)
            if feed is None:
                if self._add_feed_from_snapshot(snapshot, bar=bar):
                    updated += 1
                    continue
                skipped += 1
                continue

            if not bar:
                skipped += 1
                continue

            # Validate - skip invalid data but don't log warnings (ticker may not be trading yet)
            if not validate_ohlcv(bar):
                skipped += 1
                continue

            # Update existing feed
            feed.push_bar(bar)
            updated += 1

        elapsed = time.time() - update_start

        self._last_update_time = datetime.now(timezone.utc)
        self._symbols_updated = updated
        self._symbols_skipped = skipped

        logger.info(f"{updated} symbols updated | {skipped} skipped in {elapsed:.2f}s")
        logger.info("----------------------------------------------------------")

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dict with polling stats
        """
        return {
            "running": self._running,
            "feed_count": len(self._feeds),
            "update_count": self._update_count,
            "last_update_time": self._last_update_time,
            "symbols_updated": self._symbols_updated,
            "symbols_skipped": self._symbols_skipped,
            "api_stats": self.api_client.get_stats(),
        }

    def get_feed(self, ticker: str) -> Optional[MassiveLiveData]:
        """Get a specific data feed by ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MassiveLiveData instance or None if not found
        """
        with self._feeds_lock:
            return self._feeds.get(ticker.upper())


def setup_massive_live_feeds(
    cerebro: bt.Cerebro,
    api_key: str,
    update_interval: int = 60,
    max_symbols: Optional[int] = None,
    min_volume: Optional[float] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    auto_start: bool = True,
    use_mock: bool = False,
    mock_tickers: int = 100,
) -> MassiveFeedManager:
    """Helper function to set up live feeds from Massive API in Cerebro.

    This is the recommended way to initialize the live feed system.

    Example:
        ```python
        cerebro = bt.Cerebro()

        # Set up live feeds
        manager = setup_massive_live_feeds(
            cerebro,
            api_key="YOUR_API_KEY",
            max_symbols=100,  # Limit to 100 most active stocks
            min_volume=1_000_000,  # Filter low volume stocks
        )

        # Add strategy
        cerebro.addstrategy(MyStrategy)

        # Run live
        cerebro.run()

        # Stop polling when done
        manager.stop_polling()
        ```

    Args:
        cerebro: Backtrader Cerebro instance
        api_key: Massive.io API key (can be any string when use_mock=True)
        update_interval: Seconds between updates (default 60)
        max_symbols: Maximum number of symbols (None = all)
        min_volume: Filter stocks with prev_day volume below this
        min_price: Filter stocks with price below this
        auto_start: Automatically start polling thread
        use_mock: Use mock API for testing (default False)
        mock_tickers: Number of mock tickers to generate when using mock API (default 100)

    Returns:
        MassiveFeedManager instance (call stop_polling() when done)
    """

    # Create filter function if needed
    def symbol_filter(snapshot) -> bool:
        """Filter symbols based on volume and price criteria."""
        # Use duck typing to support both real and mock objects
        if not hasattr(snapshot, "ticker"):
            print(f"{snapshot.ticker} - Filtered lacking ticker")
            return False

        # Extract prev_day data
        prev_day = getattr(snapshot, "prev_day", None)
        if prev_day is None or not hasattr(prev_day, "close"):
            # Allow creation even without prev_day so feeds can start later
            return True

        # # Check volume filter
        # if min_volume is not None:
        #     volume = getattr(prev_day, "volume", None)
        #     if volume is None or volume < min_volume:
        #         return False

        # Check price filter
        if min_price is not None:
            close = getattr(prev_day, "close", None)
            if close is None or close < min_price:
                print(f"{snapshot.ticker} - Min price")

                return False

        return True

    # Create manager
    manager = MassiveFeedManager(
        api_key=api_key,
        update_interval=update_interval,
        max_symbols=max_symbols,
        symbol_filter=symbol_filter if (min_volume or min_price) else None,
        use_mock=use_mock,
        mock_tickers=mock_tickers,
    )

    # Create feeds
    feeds_created = manager.create_feeds(cerebro)

    if feeds_created == 0:
        logger.warning(
            "No feeds were created at startup - no trades found yet. "
            "Polling will continue and feeds will be created when data becomes available."
        )
    else:
        logger.info(f"Successfully created {feeds_created} feeds")

    # Start polling if requested
    if auto_start:
        manager.start_polling()

    return manager
