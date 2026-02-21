from ..ports.notification import NotificationService


class ResearchEngine:
    def __init__(self, notifier: NotificationService | None = None):
        self.notifier = notifier

    def run_backtest(self, config: None):
        result = "Test Result"
        print("Running backtest with config:")
        print(self.notifier)
        if self.notifier:
            print("Sending notification about backtest result...")
            self.notifier.send(f"Backtest finished. Profit: {result}")

        return result
