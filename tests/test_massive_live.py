"""Tests for Massive API live feeds."""

import time
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import aptrade as bt
import pytest

from aptrade.feeds.massive_live import (
    MassiveAPIClient,
    MassiveLiveData,
    MassiveFeedManager,
    extract_ohlcv,
    validate_ohlcv,
    setup_massive_live_feeds,
)


class MockAgg:
    """Mock Agg object from Massive API."""

    def __init__(
        self,
        o: float = 100.0,
        h: float = 105.0,
        l: float = 95.0,  # noqa: E741
        c: float = 102.0,
        v: int = 1000000,
    ):  # noqa: E741
        self.o = self.open = o
        self.h = self.high = h
        self.l = self.low = l  # noqa: E741
        self.c = self.close = c
        self.v = self.volume = v


class MockTickerSnapshot:
    """Mock TickerSnapshot from Massive API."""

    def __init__(
        self,
        ticker="AAPL",
        prev_day=None,
        min_bar=None,
        updated=None,
    ):
        self.ticker = ticker
        self.prev_day = prev_day or MockAgg()
        self.min = min_bar
        self.updated = updated or int(datetime.now(timezone.utc).timestamp() * 1e9)


def create_mock_snapshots(count=10, base_price=100.0, has_minute_data=False):
    """Create a list of mock snapshots for testing.

    Args:
        count: Number of snapshots to create
        base_price: Starting price
        has_minute_data: Whether to include minute bar data

    Returns:
        List of MockTickerSnapshot objects
    """
    snapshots = []
    for i in range(count):
        ticker = f"TICK{i:03d}"
        price = base_price + i

        prev_day = MockAgg(
            o=price,
            h=price * 1.05,
            l=price * 0.95,
            c=price * 1.02,
            v=1_000_000 + i * 100_000,
        )

        min_bar = None
        if has_minute_data:
            min_bar = MockAgg(
                o=price * 1.01,
                h=price * 1.03,
                l=price * 0.99,
                c=price * 1.015,
                v=10_000 + i * 1000,
            )

        snapshots.append(
            MockTickerSnapshot(ticker=ticker, prev_day=prev_day, min_bar=min_bar)
        )

    return snapshots


class TestExtractOHLCV:
    """Tests for extract_ohlcv function."""

    def test_extract_with_prev_day_only(self):
        """Test extraction with only prev_day data."""
        snapshot = MockTickerSnapshot("AAPL")
        result = extract_ohlcv(snapshot)

        assert result is not None
        assert result.open == 100.0
        assert result.high == 105.0
        assert result.low == 95.0
        assert result.close == 102.0
        assert result.volume == 1_000_000

    def test_extract_with_minute_data(self):
        """Test extraction prefers minute data over prev_day."""
        min_bar = MockAgg(o=101.0, h=103.0, l=99.0, c=101.5, v=50_000)
        snapshot = MockTickerSnapshot("TSLA", min_bar=min_bar)
        result = extract_ohlcv(snapshot)

        assert result is not None
        assert result.open == 101.0
        assert result.high == 103.0
        assert result.low == 99.0
        assert result.close == 101.5
        assert result.volume == 50_000

    def test_extract_invalid_snapshot(self):
        """Test extraction returns None for invalid snapshot."""
        result = extract_ohlcv("not_a_snapshot")
        assert result is None

    def test_extract_missing_prev_day(self):
        """Test extraction returns None without prev_day."""
        snapshot = MockTickerSnapshot("BAD")
        snapshot.prev_day = None
        result = extract_ohlcv(snapshot)
        assert result is None

    def test_extract_missing_ticker(self):
        """Test extraction returns None without ticker."""
        snapshot = MockTickerSnapshot("")
        result = extract_ohlcv(snapshot)
        assert result is None


class TestValidateOHLCV:
    """Tests for validate_ohlcv function."""

    def test_valid_ohlcv(self):
        """Test validation passes for valid data."""
        data = MockAgg(o=100.0, h=105.0, l=95.0, c=102.0, v=1_000_000)
        assert validate_ohlcv(data) is True

    def test_invalid_high_low(self):
        """Test validation fails when high < low."""
        data = MockAgg(o=100.0, h=95.0, l=105.0, c=102.0, v=1_000_000)  # high < low!
        assert validate_ohlcv(data) is False

    def test_invalid_negative_volume(self):
        """Test validation fails for negative volume."""
        data = MockAgg(o=100.0, h=105.0, l=95.0, c=102.0, v=-1000)
        assert validate_ohlcv(data) is False

    def test_invalid_zero_prices(self):
        """Test validation fails for zero prices."""
        data = MockAgg(o=0.0, h=105.0, l=95.0, c=102.0, v=1_000_000)
        assert validate_ohlcv(data) is False

    def test_invalid_missing_fields(self):
        """Test validation fails for missing fields."""
        data = MockAgg(o=100.0, h=None, l=None, c=None, v=None)
        assert validate_ohlcv(data) is False


class TestMassiveLiveData:
    """Tests for MassiveLiveData feed."""

    def test_initialization_with_prev_close(self):
        """Test feed initializes with prev_day_close param."""
        feed = MassiveLiveData()
        feed.p.ticker = "AAPL"
        feed.p.prev_day_close = 150.0
        assert feed.p.ticker == "AAPL"
        assert feed.p.prev_day_close == 150.0
        assert hasattr(feed, "_bar_queue")

    def test_initialization_without_prev_close(self):
        """Test feed initializes without prev_day_close."""
        feed = MassiveLiveData()
        feed.p.ticker = "TSLA"
        assert feed.p.ticker == "TSLA"
        assert hasattr(feed, "_bar_queue")

    def test_push_bar(self):
        """Test pushing new bar data."""
        feed = MassiveLiveData()
        feed.p.ticker = "AAPL"

        bar = MockAgg(
            o=100.0,
            h=105.0,
            l=95.0,
            c=102.0,
            v=1_000_000,
        )
        # Add timestamp attribute needed by the feed
        bar.timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

        feed.push_bar(bar)

        # Verify bar was queued
        assert not feed._bar_queue.empty()
        # Verify event was set
        assert feed._has_new_data.is_set()

    def test_islive(self):
        """Test feed reports as live."""
        feed = MassiveLiveData()
        feed.p.ticker = "AAPL"
        assert feed.islive() is True


class TestMassiveAPIClient:
    """Tests for MassiveAPIClient."""

    @patch("aptrade.feeds.massive_live.RESTClient")
    def test_fetch_snapshots_success(self, mock_rest_client):
        """Test successful snapshot fetch."""
        mock_client = Mock()
        mock_snapshots = create_mock_snapshots(5)
        mock_client.get_snapshot_direction.return_value = mock_snapshots
        mock_rest_client.return_value = mock_client

        client = MassiveAPIClient("test_api_key")
        result = client.fetch_snapshots()

        assert len(result) == 5
        assert result[0].ticker == "TICK000"
        mock_client.get_snapshot_direction.assert_called_once_with(
            "stocks", direction="gainers"
        )

    @patch("aptrade.feeds.massive_live.RESTClient")
    def test_fetch_snapshots_retry(self, mock_rest_client):
        """Test retry logic on failure."""
        mock_client = Mock()
        mock_client.get_snapshot_direction.side_effect = [
            Exception("API Error"),
            create_mock_snapshots(3),
        ]
        mock_rest_client.return_value = mock_client

        client = MassiveAPIClient("test_api_key", retry_delay=0.1)
        result = client.fetch_snapshots()

        assert len(result) == 3
        assert mock_client.get_snapshot_direction.call_count == 2

    @patch("aptrade.feeds.massive_live.RESTClient")
    def test_get_stats(self, mock_rest_client):
        """Test stats retrieval."""
        mock_client = Mock()
        mock_snapshots = create_mock_snapshots(10)
        mock_client.get_snapshot_direction.return_value = mock_snapshots
        mock_rest_client.return_value = mock_client

        client = MassiveAPIClient("test_api_key")
        client.fetch_snapshots()

        stats = client.get_stats()
        assert stats["snapshot_count"] == 10
        assert stats["last_fetch_time"] is not None


class TestMassiveFeedManager:
    """Tests for MassiveFeedManager."""

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_create_feeds(self, mock_api_client):
        """Test feed creation from snapshots."""
        mock_client = Mock()
        mock_snapshots = create_mock_snapshots(5)
        mock_client.fetch_snapshots.return_value = mock_snapshots
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = MassiveFeedManager("test_key")

        count = manager.create_feeds(cerebro)

        assert count == 5
        assert len(manager._feeds) == 5
        assert "TICK000" in manager._feeds

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_create_feeds_with_max_symbols(self, mock_api_client):
        """Test feed creation respects max_symbols limit."""
        mock_client = Mock()
        mock_snapshots = create_mock_snapshots(100)
        mock_client.fetch_snapshots.return_value = mock_snapshots
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = MassiveFeedManager("test_key", max_symbols=10)

        count = manager.create_feeds(cerebro)

        assert count == 10
        assert len(manager._feeds) == 10

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_create_feeds_without_prev_day(self, mock_api_client):
        """Test feeds are created even when prev_day data is missing."""
        mock_client = Mock()
        snapshot_no_prev = MockTickerSnapshot("NOPREV")
        snapshot_no_prev.prev_day = None
        snapshot_with_prev = MockTickerSnapshot("HASPREV", prev_day=MockAgg())
        mock_client.fetch_snapshots.return_value = [
            snapshot_no_prev,
            snapshot_with_prev,
        ]
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = MassiveFeedManager("test_key")

        count = manager.create_feeds(cerebro)

        assert count == 2
        assert "NOPREV" in manager._feeds
        assert "HASPREV" in manager._feeds

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_start_stop_polling(self, mock_api_client):
        """Test polling thread start/stop."""
        mock_client = Mock()
        mock_client.fetch_snapshots.return_value = []
        mock_api_client.return_value = mock_client

        manager = MassiveFeedManager("test_key", update_interval=0.1)

        assert manager._running is False

        manager.start_polling()
        assert manager._running is True

        time.sleep(0.2)  # Let it poll once

        manager.stop_polling()
        assert manager._running is False

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_get_feed(self, mock_api_client):
        """Test getting a specific feed."""
        mock_client = Mock()
        mock_snapshots = create_mock_snapshots(5)
        mock_client.fetch_snapshots.return_value = mock_snapshots
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = MassiveFeedManager("test_key")
        manager.create_feeds(cerebro)

        feed = manager.get_feed("TICK000")
        assert feed is not None
        assert feed.p.ticker == "TICK000"

        missing = manager.get_feed("MISSING")
        assert missing is None

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_adds_new_feed_on_update(self, mock_api_client):
        """Test manager adds a feed when a new symbol appears in updates."""
        mock_client = Mock()
        initial = create_mock_snapshots(1)
        update = create_mock_snapshots(1)
        update[0].ticker = "NEW01"

        mock_client.fetch_snapshots.return_value = initial
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = MassiveFeedManager("test_key")
        manager.create_feeds(cerebro)

        assert "TICK000" in manager._feeds
        assert "NEW01" not in manager._feeds

        mock_client.fetch_snapshots.return_value = update
        manager._fetch_and_update()

        assert "NEW01" in manager._feeds


class TestSetupMassiveLiveFeeds:
    """Tests for setup_massive_live_feeds helper."""

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_setup_basic(self, mock_api_client):
        """Test basic setup."""
        mock_client = Mock()
        mock_snapshots = create_mock_snapshots(10)
        mock_client.fetch_snapshots.return_value = mock_snapshots
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = setup_massive_live_feeds(
            cerebro,
            api_key="test_key",
            max_symbols=5,
            auto_start=False,
        )

        assert manager is not None
        assert len(manager._feeds) == 5
        assert manager._running is False

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_setup_with_filters(self, mock_api_client):
        """Test setup with price filters."""
        mock_client = Mock()
        # Create snapshots with varying prices
        snapshots = []
        for i in range(10):
            prev_day = MockAgg(
                o=100.0 + i * 10,
                h=105.0 + i * 10,
                l=95.0 + i * 10,
                c=100.0 + i * 10,  # Varying price: 100, 110, 120, ..., 190
                v=100_000 * (i + 1),  # Varying volume
            )
            snapshots.append(MockTickerSnapshot(f"TICK{i:03d}", prev_day=prev_day))

        mock_client.fetch_snapshots.return_value = snapshots
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = setup_massive_live_feeds(
            cerebro,
            api_key="test_key",
            min_price=150.0,  # Filter out first 6 tickers (prices 100-140)
            auto_start=False,
        )

        # Should create feeds for tickers with price >= 150 (TICK005-TICK009)
        assert len(manager._feeds) == 5

    @patch("aptrade.feeds.massive_live.MassiveAPIClient")
    def test_setup_with_filters_allows_missing_prev_day(self, mock_api_client):
        """Test setup allows feeds when prev_day is missing even with filters."""
        mock_client = Mock()
        low_price_prev = MockAgg(c=100.0)
        high_price_prev = MockAgg(c=160.0)

        snap_low = MockTickerSnapshot("LOW", prev_day=low_price_prev)
        snap_missing = MockTickerSnapshot("MISSING")
        snap_missing.prev_day = None
        snap_high = MockTickerSnapshot("HIGH", prev_day=high_price_prev)

        mock_client.fetch_snapshots.return_value = [snap_low, snap_missing, snap_high]
        mock_client.get_aggregates.return_value = []
        mock_api_client.return_value = mock_client

        cerebro = bt.Cerebro()
        manager = setup_massive_live_feeds(
            cerebro,
            api_key="test_key",
            min_price=150.0,
            auto_start=False,
        )

        assert len(manager._feeds) == 2
        assert "MISSING" in manager._feeds
        assert "HIGH" in manager._feeds


@pytest.fixture
def mock_massive_client():
    """Pytest fixture for mocked Massive API client."""
    with patch("aptrade.feeds.massive_live.RESTClient") as mock:
        mock_client = Mock()
        mock_client.get_snapshot_direction.return_value = create_mock_snapshots(10)
        mock_client.list_aggs.return_value = []
        mock.return_value = mock_client
        yield mock_client


def test_integration_with_mock(mock_massive_client):
    """Integration test with mocked API."""
    cerebro = bt.Cerebro()

    manager = setup_massive_live_feeds(
        cerebro,
        api_key="test_key",
        max_symbols=5,
        auto_start=False,
    )

    assert len(manager._feeds) == 5

    # Simulate an update
    mock_massive_client.get_snapshot_direction.return_value = create_mock_snapshots(
        5, base_price=110.0, has_minute_data=True
    )

    manager._fetch_and_update()

    stats = manager.get_stats()
    assert stats["symbols_updated"] == 5
