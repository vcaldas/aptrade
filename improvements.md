# Improvement Opportunities Under src

Scope: read-only review of `/src` focused on expensive loops, generator opportunities, memory inefficiencies, and code-footprint reductions.

## 1) Frequent `copy.copy(trade)` in execution loop
- Location: `src/aptrade/strategy.py` lines 588, 590, 616, 618, 621, 623
- Code to revisit: repeated `copy.copy(trade)` appends to `_tradespending` and `qtrades` inside `for exbit in order.executed.iterpending()`.
- Why improve: this is on a hot path during order processing; repeated object copying increases allocation churn and GC pressure.
- Improvement direction: reduce number of copies by consolidating notification paths, or use immutable/compact trade snapshots for pending notifications.

## 2) Notification queues are fully materialized into lists
- Location: `src/aptrade/store.py` line 92
- Location: `src/aptrade/stores/vcstore.py` line 397
- Location: `src/aptrade/stores/oandastore.py` line 271
- Code to revisit: `return [x for x in iter(self.notifs.popleft, None)]`
- Why improve: forces full list allocation even when caller may iterate once.
- Improvement direction: return an iterator/generator and materialize only where truly needed.

## 3) Multiple DataFrame `.copy()` calls during date filtering
- Location: `src/aptrade/feeds/pandafeed.py` lines 376, 380, 386, 390
- Code to revisit: each date bound filter applies `.copy()` independently.
- Why improve: each copy duplicates potentially large dataframes; chained filtering causes avoidable memory spikes.
- Improvement direction: apply combined mask first, then perform at most one copy (or avoid copy if safe).

## 4) Repeated append/pop loops for bulk buffer resizing
- Location: `src/aptrade/linebuffer.py` lines 252, 265, 292
- Code to revisit: `for i in range(size): self.array.append(value)` and `for i in range(size): self.array.pop()`.
- Why improve: Python loop overhead is high for bulk operations.
- Improvement direction: use vectorized list ops (`extend`, slice deletion) for batch resize operations.

## 5) `locals().copy()` used to build thread kwargs
- Location: `src/aptrade/stores/ibstore.py` line 686
- Location: `src/aptrade/stores/vcstore.py` line 504
- Location: `src/aptrade/stores/oandastore.py` line 366
- Code to revisit: `kwargs = locals().copy()` then mutate.
- Why improve: copies full local scope (including intermediates) and increases memory footprint/noise.
- Improvement direction: explicitly construct kwargs dict with only required keys.

## 6) Repeated list comprehensions over `datas` in `_runnext`
- Location: `src/aptrade/cerebro.py` lines 1560, 1561, 1562
- Code to revisit:
  - `rs = [i for i, x in enumerate(datas) if x.resampling]`
  - `rp = [i for i, x in enumerate(datas) if x.replaying]`
  - `rsonly = [i for i, x in enumerate(datas) if x.resampling and not x.replaying]`
- Why improve: three full passes and three list allocations over the same collection.
- Improvement direction: compute all three in one pass, or use lazy sets/iterators where random access is unnecessary.

## 7) `max([...])` with temporary list allocation
- Location: `src/aptrade/lineiterator.py` lines 119, 134
- Code to revisit:
  - `max([x._minperiod for x in _obj.datas] or [_obj._minperiod])`
  - `max([x._minperiod for x in _obj.lines])`
- Why improve: creates temporary lists for simple reductions.
- Improvement direction: switch to generator expressions and default handling.

## 8) Indicator filtering returns fully materialized list
- Location: `src/aptrade/lineiterator.py` line 209
- Code to revisit: `return [x for x in self._ind_iterator if hasattr(x.lines, "getlinealiases")]`
- Why improve: full list allocation every call, even if caller only iterates once.
- Improvement direction: return iterator/generator, or cache filtered set if repeatedly requested.

## 9) Order book helpers eagerly clone/materialize lists
- Location: `src/aptrade/brokers/bbroker.py` lines 501, 503, 542
- Code to revisit:
  - `os = [x.clone() for x in self.pending]`
  - `os = [x for x in self.pending]`
  - `rets = [self.transmit(x, check=check) for x in pc]`
- Why improve: avoids streaming behavior and allocates full intermediate lists.
- Improvement direction: iterate directly where possible; only build list when caller requires random access.

## 10) Redundant numeric conversion
- Location: `src/aptrade/feeds/csvgeneric.py` line 152
- Code to revisit: `line[0] = float(float(csvfield))`
- Why improve: duplicated conversion does unnecessary work on every parsed record.
- Improvement direction: convert once; validate/normalize upstream if needed.

## 11) High-frequency timestamp calls inside data loop
- Location: `src/aptrade/cerebro.py` lines 1593, 1595
- Code to revisit: `datetime.datetime.now(datetime.UTC)` called for each data in each loop iteration.
- Why improve: system clock calls are relatively expensive in tight loops.
- Improvement direction: minimize repeated `now()` calls (reuse monotonic timing deltas or batch timing calculation).

## 12) Per-bar file reads in binary feeds
- Location: `src/aptrade/feeds/vchart.py` line 95
- Location: `src/aptrade/feeds/vchartfile.py` line 100
- Code to revisit: one `self.f.read(...)` per bar in `_load()`.
- Why improve: high syscall overhead on large datasets.
- Improvement direction: buffered/chunk reads or memory-mapped access to reduce read-call frequency.

## 13) Large intermediate list creation when building equity rows
- Location: `src/aptrade/analyzers/eq.py` line 158
- Code to revisit: `data = [[k] + v[-2:] for k, v in iteritems(self.rets)]`
- Why improve: builds full in-memory nested list before `DataFrame.from_records`.
- Improvement direction: stream rows via generator/iterator, or construct DataFrame directly from dict structures.

## 14) Unused resampled series allocation
- Location: `src/aptrade/analyzers/eq.py` line 215
- Code to revisit: `day_eq = eq_df["value"].resample("D").last().dropna()` (assigned but not used).
- Why improve: unnecessary compute + memory allocation on larger equity curves.
- Improvement direction: remove dead computation or use it in downstream metrics.

## 15) Verbose string assembly in `Order.__str__`
- Location: `src/aptrade/order.py` lines 376-392
- Code to revisit: 17 repeated `tojoin.append("...{}".format(...))` calls.
- Why improve: high object churn when string conversion happens frequently (logging/debug-heavy runs).
- Improvement direction: use a compact tuple-driven formatter or lazy logging to reduce repeated string formatting work.

---

## Suggested Prioritization
1. High impact first: items 1, 2, 3, 4, 6, 12.
2. Medium impact: items 5, 7, 8, 9, 11, 13.
3. Low risk/code-footprint cleanup: items 10, 14, 15.

## Notes
- No code was modified for this review.
- Recommendations prioritize runtime/memory behavior in backtest hot paths and high-volume data processing sections.
