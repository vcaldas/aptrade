# Feed Module Improvement Opportunities

Scope: read-only evaluation of feed infrastructure and concrete feed
implementations under `src/aptrade/feeds`, with emphasis on reducing duplication
and improving runtime/memory efficiency before adding new feeds.

## Executive Summary

The feed layer has a strong base abstraction in `feed.py`, but concrete
implementations repeat the same patterns in several places:

- Live/historical finite-state-machine logic is duplicated across broker-backed
  feeds.
- OHLC assignment and datetime conversion are repeated across many `_load`
  methods.
- Several modules duplicate file/open/read/close lifecycle logic.
- Pandas feed variants repeat mapping/filtering/load logic with small
  differences.
- Some hot paths allocate avoidable intermediate lists.

The fastest path to safer extension is to standardize shared building blocks
first (state machine helpers, row/bar mappers, and reusable adapters).

---

## 1) Duplicate live-feed state machine logic (High Impact)

### Evidence

- `src/aptrade/feeds/ibdata.py` line 216:
  `_ST_FROM, _ST_START, _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(5)`
- `src/aptrade/feeds/oanda.py` line 156: same state constants and overall flow
- `_load` loops implementing state transitions:
  - `src/aptrade/feeds/ibdata.py` line 444
  - `src/aptrade/feeds/oanda.py` line 263

### Why this matters

Two independent implementations of nearly the same state machine increase
maintenance cost and drift risk (bug fixes, reconnect behavior, status
transitions, backfill edge cases).

### Improvement direction

Define a shared live-feed orchestration layer with overridable hooks:

- fetch live message
- fetch historical/backfill message
- parse transport error codes
- map message to bar
- reconnect policy

Keep provider-specific transport details in thin subclasses.

---

## 2) Duplicate `backfill_from` line-copy block (High Impact)

### Evidence

- `src/aptrade/feeds/ibdata.py` line 638
- `src/aptrade/feeds/oanda.py` line 393

Both copy every line alias from one feed to another in the same style.

### Why this matters

Duplicated synchronization logic is easy to subtly diverge (line alias changes,
future line additions, edge behavior).

### Improvement direction

Move this behavior into a shared `DataBase` helper (for example, a reusable
"copy current bar from source" utility) and reuse from all backfill-capable
feeds.

---

## 3) Repeated OHLC assignment blocks in many `_load` methods (High Impact)

### Evidence

- `src/aptrade/feeds/vchart.py` lines 116-121
- `src/aptrade/feeds/vchartfile.py` lines 133-138
- `src/aptrade/feeds/vcdata.py` lines 480-485
- `src/aptrade/feeds/rollover.py` lines 194-199
- `src/aptrade/feeds/ibdata.py` lines 700-705 and 722-727
- `src/aptrade/feeds/oanda.py` lines 419-424 and 442-455

### Why this matters

Same field-setting pattern is replicated with minor variations. This increases
bug surface and code size.

### Improvement direction

Standardize line population through internal helpers:

- set OHLCV+OI from tuple/sequence
- set OHLC from scalar tick
- set from provider field map

This improves consistency and makes new feeds easier to implement.

---

## 4) Duplicate binary VisualChart parsing logic (High Impact)

### Evidence

- Binary date decode appears in both files:
  - `src/aptrade/feeds/vchart.py` lines 102, 104
  - `src/aptrade/feeds/vchartfile.py` lines 116, 117
- Repeated file lifecycle/read logic:
  - `src/aptrade/feeds/vchart.py` lines 83, 87, 95
  - `src/aptrade/feeds/vchartfile.py` lines 86, 92, 100

### Why this matters

Both modules implement nearly identical low-level unpack/decode logic and
per-bar IO handling.

### Improvement direction

Create a shared binary adapter for:

- bar format metadata selection
- record unpacking
- packed date/time decode
- file lifecycle and guarded read

Then keep only path/store specifics in each feed.

---

## 5) Pandas mapping/loading logic duplicated in `PandasData` and `PandasDataNew` (High Impact)

### Evidence

- duplicate mapping setup:
  - `src/aptrade/feeds/pandafeed.py` lines 172-175
  - `src/aptrade/feeds/pandafeed.py` lines 342-345
- duplicate column normalization:
  - `src/aptrade/feeds/pandafeed.py` line 207
  - `src/aptrade/feeds/pandafeed.py` line 404
- duplicate row load loops:
  - `src/aptrade/feeds/pandafeed.py` line 237
  - `src/aptrade/feeds/pandafeed.py` line 443

### Why this matters

Two classes with mostly shared behavior increase maintenance burden and create
optimization inconsistency.

### Improvement direction

Consolidate shared dataframe adapter logic into one base utility:

- build/normalize `_colmapping`
- convert datetime source (index vs column)
- row extraction strategy (iterative vs preloaded)

Then use strategy flags for preloading/filter behavior.

---

## 6) Dataframe filtering in `PandasDataNew` copies multiple times (Performance + Memory)

### Evidence

- `src/aptrade/feeds/pandafeed.py` lines 376, 380, 386, 390 call `.copy()`
  repeatedly.

### Why this matters

Each copy duplicates potentially large dataframes and increases memory pressure.

### Improvement direction

Build one combined filter mask and apply a single materialization step.

---

## 7) Row-adapter duplication between `PandasDirectData` and `BlazeData` (Medium Impact)

### Evidence

- `src/aptrade/feeds/pandafeed.py` line 31: `PandasDirectData`
- `src/aptrade/feeds/blaze.py` line 27: `BlazeData`

Both perform:

- iterator startup in `start`
- per-row field mapping in `_load`
- datetime conversion to numeric format

### Why this matters

Repeated adapter shape increases code footprint and slows introduction of new
row-based feeds.

### Improvement direction

Extract a generic row-iterator adapter base (row source + schema mapping +
datetime extractor hook).

---

## 8) CSV parsing hot path does manual split/tokenization (Medium Impact)

### Evidence

- `src/aptrade/feed.py` lines 719/725 and 733/739 (`readline` + `split`)
- date/time parsing + float conversions are repeated in multiple CSV feeds.

### Why this matters

Manual tokenization works but leaves performance and robustness opportunities on
the table (quoting, edge formatting, parser centralization).

### Improvement direction

Centralize parsing/token coercion policy once for CSV feeds (including float
coercion/null policy and datetime parser caching).

---

## 9) Repeated dynamic tick-attribute assignment in base feed (Medium Impact)

### Evidence

- `src/aptrade/feed.py` lines 365-377 (`setattr` per alias in `_tick_nullify`
  and `_tick_fill`)

### Why this matters

Repeated dynamic attribute lookup/assignment in high-frequency paths adds
overhead.

### Improvement direction

Cache alias references once and use indexed/struct-like storage for tick fields
in hot path operations.

---

## 10) Stack save path allocates full list per bar (Medium Impact)

### Evidence

- `src/aptrade/feed.py` line 564: `bar = [line[0] for line in self.itersize()]`

### Why this matters

This allocation appears in stack operations and can be expensive under heavy
filtering/resampling.

### Improvement direction

Use reusable buffer structures or tuple snapshots where mutability is not
needed.

---

## 11) Yahoo reverse mode loads full file in memory (Medium Impact)

### Evidence

- `src/aptrade/feeds/yahoo.py` lines 99, 101, 104 use deque + reverse rebuild.

### Why this matters

For large files, full in-memory reversal has high peak memory cost.

### Improvement direction

Use streaming reverse strategy or bounded chunk approach when file size exceeds
threshold.

---

## 12) Influx feed materializes all points before iteration (Medium Impact)

### Evidence

- `src/aptrade/feeds/influxfeed.py` line 109:
  `dbars = list(self.ndb.query(qstr).get_points())`

### Why this matters

Materializing entire query result upfront increases latency and memory footprint
for large result sets.

### Improvement direction

Iterate lazily over query cursor/generator and stream bars directly into
`_load`.

---

## 13) Package import strategy eagerly loads many feed modules (Low-Medium Impact)

### Evidence

- Wildcard imports in `src/aptrade/feeds/__init__.py` lines 25-33, 35, 40, 56.

### Why this matters

Eager imports increase import-time cost and may force optional dependency checks
early.

### Improvement direction

Adopt explicit exports and lazy optional-feed registration to reduce startup
overhead and import side effects.

---

## 14) Meta registration pattern repeated across store-backed feeds (Low-Medium Impact)

### Evidence

- `MetaIBData`: `src/aptrade/feeds/ibdata.py` line 33
- `MetaOandaData`: `src/aptrade/feeds/oanda.py` line 32
- `MetaVCData`: `src/aptrade/feeds/vcdata.py` line 33
- `MetaVChartFile`: `src/aptrade/feeds/vchartfile.py` line 31

Each does class registration to a store DataCls target in a similar way.

### Why this matters

Boilerplate repetition and room for inconsistent registration behavior.

### Improvement direction

Centralize store-registration metaclass helper or declarative registration hook.

---

## 15) Stale duplicated commented code in massivefeed (Code Footprint)

### Evidence

- `src/aptrade/feeds/massivefeed.py` line 35 (`# class MassiveCSVData...`)
- `src/aptrade/feeds/massivefeed.py` line 194 (`# class YahooFinanceData...`)

### Why this matters

Large commented-out duplicated implementation increases noise and maintenance
burden.

### Improvement direction

Move historical reference implementations to docs/changelog and keep production
modules minimal.

---

## 16) Minor but repeated conversion inefficiency in CSV generic feed

### Evidence

- `src/aptrade/feeds/csvgeneric.py` line 152: `line[0] = float(float(csvfield))`

### Why this matters

Small but hot-path overhead in row parsing.

### Improvement direction

Normalize conversion path once per field assignment.

---

## 17) New-feed development ergonomics: no canonical feed skeleton (Process + Quality)

### Evidence

Multiple feed styles exist (CSV, dataframe, binary, store/live FSM) with no
single “authoritative” skeleton for new implementations.

### Why this matters

New feed authors are likely to duplicate whichever existing file they copy
first, perpetuating divergence.

### Improvement direction

Define a documented internal feed template matrix:

- pull batch (CSV/binary)
- row-iterator adapter
- live+historical store-backed FSM
- combinator feed (chainer/rollover)

Include checklist for required methods, notification semantics, and performance
guardrails.

---

## Prioritized Refactor Plan (No Code Yet)

1. Standardize live FSM helper and migrate `ibdata` + `oanda` first.
2. Extract reusable OHLC assignment and bar-copy utilities in base feed layer.
3. Consolidate pandas adapters into one shared mapping engine.
4. Unify VisualChart binary parsers.
5. Introduce streaming/lazy handling for Influx and Yahoo reverse mode.
6. Reduce module footprint (`massivefeed` cleanup and import strategy review).

---

## Suggested Acceptance Criteria Before Implementing New Feed

- One shared live-feed orchestration abstraction exists.
- One shared field-assignment utility exists for OHLCV/OI payloads.
- One shared row-mapping utility exists for dataframe/row feeds.
- New feed template/checklist is documented and used for all additions.
- Benchmarks include memory peak + throughput for large source datasets.

---

## Notes

- This document is read-only analysis; no production code was modified.
- Recommendations favor maintainability first, then performance, then footprint
  reduction.
