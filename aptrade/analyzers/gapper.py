import aptrade as bt


class Gapper(bt.Analyser):
    """Classify instruments by the gap between today's high and yesterday's close."""

    params = dict(threshold=1.0, output_dir="./gapper_results", save_csv=True)

    def start(self):
        self.events = []

    def next(self):
        for data in self.datas:
            if len(data) < 2:
                continue
            current_high = data.high[0]
            previous_close = data.close[-1]
            gap = (current_high - previous_close) / previous_close
            if gap >= self.p.threshold:
                when = data.datetime.datetime(0)
                self.events.append(
                    (data._name, when, current_high, previous_close, gap)
                )

    def stop(self):
        self.rets["over"] = self.events

        if not self.params.save_csv or not self.events:
            return

        import os

        import pandas as pd

        output_dir = self.params.output_dir
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "gapper_events.csv")

        df = pd.DataFrame(
            self.events,
            columns=[
                "Ticker",
                "Timestamp",
                "Current High",
                "Previous Close",
                "Gap (%)",
            ],
        )

        if os.path.exists(output_path):
            df.to_csv(output_path, mode="a", header=False, index=False)
        else:
            df.to_csv(output_path, index=False)
