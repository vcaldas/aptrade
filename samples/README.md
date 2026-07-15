# Samples

## Reference Workflow

Use this process for each missing sample reference:

1. Identify the script that pytest will execute. The reference file must be
   named `<script-stem>.txt` and live next to the script.

2. Prefer a deterministic command. Use `uv run python ...` and prefer
   local/offline data over live network calls when the sample supports it.

3. If the sample needs non-default CLI arguments to produce meaningful stdout,
   add them to `SAMPLE_ARGS` in `tests/samples/test_samples.py`. Do not create
   empty reference files for samples that should emit output.

4. Generate the reference from the exact command pytest will use. Example
   pattern: `uv run python path/to/sample.py [args...] > path/to/sample.txt`

5. Update this table. Mark the sample as covered and describe any special
   invocation in the Comment column.

6. Run a narrow validation. Use
   `uv run pytest tests/samples/test_samples.py -k <sample-name>` before moving
   to the next file.

7. If a sample still produces no stdout by design, do not add a blank file
   silently. First confirm that no-output behavior is intentional and note it
   explicitly in the Comment column.

| Filename              | Covered | Comment                                                                                                            |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| analyzer-annualreturn | ✅      |                                                                                                                    |
| bidask-to-ohlc        | ✅      |                                                                                                                    |
| bracket               | ✅      |                                                                                                                    |
| btfd                  | ✅      | Offline closeclose run with prorder/prdata reference                                                               |
| calendar-days         | ✅      | Writer CSV output with default data                                                                                |
| calmar                | ✅      |                                                                                                                    |
| cheat-on-open         | ✅      |                                                                                                                    |
| commission-schemes    | ✅      |                                                                                                                    |
| credit-interest       | ✅      |                                                                                                                    |
| data-bid-ask          | ✅      |                                                                                                                    |
| data-filler           | ✅      | Writer CSV output over a narrowed intraday session with filler enabled                                             |
| data-parquet          | ❌      | Blocked: no parquet datastore fixtures are present in the repo                                                     |
| data-multitimeframe   | ✅      | Indicator output using weekly input resampled to monthly                                                           |
| data-pandas           | ✅      | `data-pandas.py` and `data-pandas-optix.py` references with plotting disabled                                      |
| data-replay           | ✅      | Weekly replay output using repo-root daily input path                                                              |
| data-resample         | ✅      | Smoke test with weekly resample and plotting disabled                                                              |
| daysteps              | ✅      |                                                                                                                    |
| future-spot           | ✅      | Smoke test with repo-root daily input, fixed seed, and plotting disabled                                           |
| gold-vs-sp500         | ✅      | Smoke test using offline local Yahoo CSV inputs                                                                    |
| ib-cash-bid-ask       | ❌      |                                                                                                                    |
| ibtest                | ❌      |                                                                                                                    |
| kselrsi               | ✅      | Offline Yahoo CSV window in 2013 with order-event reference output                                                 |
| lineplotter           | ✅      | Smoke-only plotting demo with repo-root local data path                                                            |
| lrsi                  | ✅      | Smoke-only indicator demo with repo-root local data path                                                           |
| macd-settings         | ✅      | Analyzer output with explicit repo-root Yahoo CSV path instead of stale dataset shortcut                           |
| massive-test          | ✅      | Writer output with offline Yahoo CSV input over a narrowed daily window                                            |
| memory-savings        | ✅      | Total memory-cell count now covered by `memory-savings.txt`                                                        |
| mixing-timeframes     | ✅      | Pivot-point coupling output with repo-root daily input path                                                        |
| multi-copy            | ✅      | Short copied-data window in `runnext` mode to avoid batch-path failure                                             |
| multi-example         | ✅      | Multi-data bracket output with repo-root Yahoo CSV inputs                                                          |
| multidata-strategy    | ✅      | Aligned and unaligned multi-data outputs covered with narrowed offline windows                                     |
| multitrades           | ✅      | `multitrades.py` reference plus smoke coverage for helper `mtradeobserver.py`                                      |
| oandatest             | ❌      |                                                                                                                    |
| observer-benchmark    | ✅      | Printout mode with repo-root Yahoo CSV inputs                                                                      |
| observers             | ✅      | Drawdown and orderobserver references plus smoke tests for plot-only/helper modules                                |
| oco                   | ✅      |                                                                                                                    |
| optimization          | ✅      | Single-combination deterministic optimization run with local daily input                                           |
| order-close           | ✅      | close-daily uses fixed seed and both references use repo-root data paths                                           |
| order-execution       | ✅      | Market execution with repo-root daily input path                                                                   |
| order-history         | ✅      | Signal strategy output with repo-root daily input path                                                             |
| order_target          | ✅      | Target-size mode with repo-root Yahoo CSV input path                                                               |
| partial-plot          | ✅      | Smoke-only partial-plot demo with repo-root daily data path                                                        |
| pinkfish-challenge    | ✅      |                                                                                                                    |
| pivot-point           | ✅      | `ppsample.py` reference with narrowed daily window plus smoke coverage for helper `pivotpoint.py`                  |
| plot-same-axis        | ✅      | Smoke-only plotting demo with explicit `--noplot` to avoid optional chart dependency                               |
| psar                  | ✅      | Daily and intraday PSAR outputs covered with narrowed offline windows                                              |
| pyfolio2              | ❌      |                                                                                                                    |
| pyfoliotest           | ❌      |                                                                                                                    |
| relative-volume       | ✅      | `relative-volume.py` writer output with local intraday volume data plus smoke coverage for helper `relvolbybar.py` |
| renko                 | ✅      | Smoke-only Renko filter demo using `runonce=False` to avoid batch-path failure                                     |
| resample-tickdata     | ✅      | Local tick-sample writer output with explicit `--noplot`                                                           |
| rollover              | ❌      |                                                                                                                    |
| sharpe-timereturn     | ✅      | Writer/analyzer output with narrowed daily window                                                                  |
| signals-strategy      | ✅      | Smoke-only signal framework sample after Python 3 CLI fix                                                          |
| sigsmacross           | ✅      | `sigsmacross.py` trade-output reference plus smoke coverage for `sigsmacross2.py`                                  |
| sizertest             | ✅      | Smoke-only sizer sample after legacy sizer API compatibility fixes                                                 |
| slippage              | ✅      | Trade-output reference with narrowed daily window                                                                  |
| sratio                | ✅      | Default Sharpe ratio calculation now covered by `sratio.txt`                                                       |
| stop-trading          | ✅      | Manual stop-loss approach with repo-root daily input path                                                          |
| stoptrail             | ✅      | Default trailing-stop output with repo-root daily input path                                                       |
| strategy-selection    | ✅      | Strategy optimization selection output now covered by `strategy-selection.txt`                                     |
| talib                 | ✅      | Smoke coverage only; optional TA-Lib bridge is unavailable in this environment                                     |
| timers                | ✅      | `scheduled.py` and `scheduled-min.py` references from narrowed timer windows                                       |
| tradingcalendar       | ✅      | `tcal.py` and `tcal-intra.py` references with local offline data and fixed timezone/calendar behavior              |
| vctest                | ❌      |                                                                                                                    |
| volumefilling         | ✅      | Default volume-filling output now covered by `volumefilling.txt`                                                   |
| vwr                   | ✅      | Default analyzer/writer output now covered by `vwr.txt`                                                            |
| weekdays-filler       | ✅      | `weekdaysaligner.py` reference with offline Yahoo CSV inputs plus smoke coverage for `weekdaysfiller.py`           |
| writer-test           | ✅      | Default writer/analyzer output now covered by `writer-test.txt`                                                    |
| yahoo-test            | ❌      |                                                                                                                    |

# Missing

- samples/data-parquet/parquetgeneric.py
- samples/ib-cash-bid-ask/ib-cash-bid-ask.py
- samples/ibtest/ibtest.py
- samples/oandatest/oandatest.py
- samples/pinkfish-challenge/pinkfish-challenge.py
- samples/pyfolio2/pyfoliotest.py
- samples/pyfoliotest/sample_pyfoliotest.py
- samples/rollover/rollover.py
- samples/vctest/vctest.py
- samples/yahoo-test/yahoo-test.py
