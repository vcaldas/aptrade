# from datetime import datetime, time

# from aptrade.domain.models.market_event import MarketEvent
# from aptrade.domain.scanner.service import Scanner, ScannerConfig


# class FakeDataPort:
#     def __init__(self, events):
#         self._events = list(events)

#     def events(self):
#         yield from self._events

#     def close(self):
#         return None


# class FakeNotifier:
#     def __init__(self):
#         self.messages = []

#     def notify(self, title: str, body: str) -> None:
#         self.messages.append((title, body))


# def test_scanner_filters_and_notify():
#     events = [
#         MarketEvent(symbol="AAA", timestamp=datetime(2026, 2, 21, 9, 0), price=10.0, volume=1_000_000),
#         MarketEvent(symbol="BBB", timestamp=datetime(2026, 2, 21, 9, 1), price=1.5, volume=100),
#     ]

#     notifier = FakeNotifier()
#     cfg = ScannerConfig(max_symbols=10, min_volume=500_000, min_price=None, max_price=50.0)
#     scanner = Scanner(data_port=FakeDataPort(events), notifier=notifier)

#     scanner.run(cfg)

#     assert len(notifier.messages) == 1
#     assert "AAA" in notifier.messages[0][0]


# def test_scanner_time_window():
#     events = [
#         MarketEvent(symbol="EARLY", timestamp=datetime(2026, 2, 21, 3, 0), price=5.0, volume=1_000_000),
#         MarketEvent(symbol="OPEN", timestamp=datetime(2026, 2, 21, 9, 30), price=7.0, volume=1_000_000),
#     ]

#     notifier = FakeNotifier()
#     cfg = ScannerConfig(max_symbols=10, min_volume=0, min_price=None, max_price=None, start_time=time(4, 0), stop_time=time(20, 0))
#     scanner = Scanner(data_port=FakeDataPort(events), notifier=notifier)

#     scanner.run(cfg)

#     # Only the OPEN event falls within the configured time window
#     assert len(notifier.messages) == 1
#     assert "OPEN" in notifier.messages[0][0]
