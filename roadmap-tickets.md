# Trading System Epics and Tickets

This file breaks the roadmap into epics and issue-sized tickets. Each item is
written so it can be copied into GitHub issues with minimal editing.

## GitHub Labels

- `epic:foundation`
- `epic:market-data`
- `epic:brokers`
- `epic:strategy-runtime`
- `epic:risk-controls`
- `epic:operations`
- `epic:scheduling`
- `area:frontend`
- `area:backend`
- `area:aptrade`
- `area:airflow`
- `area:celery`
- `area:celery-beat`
- `area:infra`
- `type:epic`
- `type:feature`
- `type:integration`
- `type:api`
- `type:ui`
- `type:worker`
- `type:pipeline`
- `type:ops`
- `priority:p0`
- `priority:p1`
- `priority:p2`
- `priority:p3`

## GitHub Milestones

- `M1 Foundations`: contracts, canonical models, core architecture decisions
- `M2 Data Baseline`: market-data adapters, backend data APIs, validation UI
- `M3 Broker Baseline`: broker adapters, broker APIs, broker operations UI
- `M4 Scheduling Core`: Airflow, Celery, Celery Beat, task status APIs
- `M5 Strategy Paper Mode`: strategy runtime, paper execution, audit trail
- `M6 Safety and Ops`: risk controls, observability, deployment, runbooks

## Strict Implementation Sequence

Build in this order. A later ticket should not start until its required
predecessors are complete.

1. `1.1` Define canonical trading domain models
2. `1.2` Define broker adapter interface
3. `1.3` Define market-data adapter interface
4. `1.4` Create provider and broker capability matrix
5. `1.5` Standardize logging, tracing, and correlation IDs
6. `1.6` Resolve core architecture decisions
7. `7.1` Define task ownership by execution system
8. `7.2` Add Airflow infrastructure for daily pipelines
9. `7.4` Add Celery worker infrastructure for on-demand jobs
10. `7.6` Add Celery Beat for minute-level recurring jobs
11. `7.8` Add backend task submission and status APIs
12. `2.3` Create symbol normalization and lookup rules
13. `2.1` Implement Massive market-data adapter
14. `2.4` Add backend market-data service layer
15. `2.5` Add market-data API endpoints
16. `2.6` Build frontend market-data validation screen
17. `7.3` Build daily data pipeline DAG in Airflow
18. `7.7` Implement minute polling task
19. `3.1` Implement Interactive Brokers adapter
20. `3.3` Normalize broker order-state machine
21. `3.4` Add backend broker service layer
22. `3.5` Add backend broker API endpoints
23. `3.6` Build frontend broker operations screen
24. `3.7` Add live-trading safety flag and permissions gate
25. `7.5` Implement backtest job as first Celery task
26. `7.9` Build frontend task operations screen
27. `7.10` Add observability for schedulers and workers
28. `2.7` Add provider failover rules
29. `2.2` Implement BarChart market-data adapter
30. `3.2` Implement TradeZero adapter
31. `4.1` Define strategy lifecycle and state model
32. `4.2` Define strategy configuration contract
33. `4.3` Implement backend strategy orchestration service
34. `4.4` Add paper-trading execution path
35. `4.5` Build frontend strategy management screens
36. `4.6` Add strategy audit trail and event log
37. `5.1` Implement pre-trade risk engine
38. `5.2` Add emergency stop and circuit breakers
39. `5.3` Add health checks and heartbeats
40. `5.4` Implement reconnect and stale-state handling
41. `5.5` Add alerting for critical failures
42. `5.6` Add event replay and debugging support
43. `6.1` Build integration test suite for adapters
44. `6.2` Add paper-trading regression flows
45. `6.3` Define deployment profiles
46. `6.4` Implement secrets and credential management
47. `6.5` Write operational runbooks
48. `6.6` Define metrics and dashboard requirements

## Epic 1: Foundation and Contracts

**Description**

Create the shared architecture contracts that keep vendor-specific logic inside
`aptrade` and let the backend and frontend depend on stable internal models.

**Epic Checklist**

- [ ] Canonical domain models are defined.
- [ ] Broker and market-data adapter interfaces are defined.
- [ ] Capability matrix structure exists for brokers and providers.
- [ ] Logging and correlation standards are defined.
- [ ] Core architecture decisions are documented.

### [ ] Ticket 1.1: Define canonical trading domain models

**Description**

Create normalized models for instruments, quotes, bars, orders, fills,
positions, balances, account summaries, strategy inputs, and strategy events so
the rest of the platform can remain vendor-agnostic.

**Checklist**

- [ ] Define required and optional fields for `Instrument`, `Quote`, and `Bar`.
- [ ] Define required and optional fields for `OrderRequest`, `OrderStatus`, and
      `ExecutionEvent`.
- [ ] Define required and optional fields for `PositionSnapshot`,
      `BalanceSnapshot`, and `AccountSummary`.
- [ ] Define timestamp, timezone, and session conventions.
- [ ] Document the models in a shared reference file.

### [ ] Ticket 1.2: Define broker adapter interface

**Description**

Create a `BrokerAdapter` contract in `aptrade` that standardizes account access,
order submission, order cancellation, order status updates, positions, and
connection health.

**Checklist**

- [ ] Define broker connection lifecycle methods.
- [ ] Define account summary and positions retrieval methods.
- [ ] Define place, cancel, and modify order methods.
- [ ] Define normalized order status and execution event callbacks.
- [ ] Define broker capability reporting.

### [ ] Ticket 1.3: Define market-data adapter interface

**Description**

Create a `MarketDataAdapter` contract in `aptrade` that standardizes latest
quotes, historical bars, subscriptions, provider health, and fallback behavior.

**Checklist**

- [ ] Define methods for latest quote retrieval.
- [ ] Define methods for historical bar retrieval.
- [ ] Define subscription and unsubscribe behavior where supported.
- [ ] Define provider health reporting.
- [ ] Define provider capability reporting.

### [ ] Ticket 1.4: Create provider and broker capability matrix

**Description**

Define a single capability schema that captures supported order types,
historical data coverage, real-time behavior, paper-trading support,
extended-hours support, and known constraints.

**Checklist**

- [ ] Define fields for broker capabilities.
- [ ] Define fields for data-provider capabilities.
- [ ] Add placeholders for Interactive Brokers, TradeZero, Massive, and
      BarChart.
- [ ] Document unsupported or unknown capabilities explicitly.
- [ ] Make the capability schema accessible to backend services.

### [ ] Ticket 1.5: Standardize logging, tracing, and correlation IDs

**Description**

Define the logging structure used across frontend, backend, and `aptrade` so
requests, market-data events, and order events can be traced end to end.

**Checklist**

- [ ] Define correlation ID propagation rules.
- [ ] Define structured log fields for requests, broker events, and
      data-provider events.
- [ ] Define log severity conventions.
- [ ] Define error classification categories.
- [ ] Document the standard for future implementation.

### [ ] Ticket 1.6: Resolve core architecture decisions

**Description**

Document and validate the current architecture defaults so implementation can
proceed without reopening the same decisions in every ticket.

**Checklist**

- [ ] Document PostgreSQL as the primary database and event-storage baseline.
- [ ] Document the default split across Airflow, Celery, and Celery Beat.
- [ ] Document live strategy orchestration in the backend as the first-release
      default.
- [ ] Document `aptrade` as an internal Python package for the first release.
- [ ] Document uppercase canonical symbols with exchange and asset metadata
      stored separately.
- [ ] Document WebSockets with polling fallback as the UI update baseline.

## Epic 2: Market Data Baseline

**Description**

Deliver normalized market-data access through the backend and frontend, starting
with Massive and then adding BarChart as an alternate or fallback provider.

**Epic Checklist**

- [ ] Massive works end to end.
- [ ] BarChart works end to end.
- [ ] Backend serves normalized quote and bar data.
- [ ] Frontend can validate symbols and display provider-backed data.
- [ ] Provider health and failover state are visible.

### [ ] Ticket 2.1: Implement Massive market-data adapter

**Description**

Build the initial Massive integration in `aptrade` for latest quotes, historical
bars, provider health, and normalized symbol handling.

**Checklist**

- [ ] Add Massive configuration and credentials handling.
- [ ] Implement latest quote retrieval.
- [ ] Implement historical bar retrieval.
- [ ] Normalize timestamps, intervals, and sessions.
- [ ] Expose provider health status.

### [ ] Ticket 2.2: Implement BarChart market-data adapter

**Description**

Build the BarChart integration in `aptrade` using the same normalized contract
as Massive.

**Checklist**

- [ ] Add BarChart configuration and credentials handling.
- [ ] Implement latest quote retrieval.
- [ ] Implement historical bar retrieval.
- [ ] Normalize timestamps, intervals, and sessions.
- [ ] Expose provider health status.

### [ ] Ticket 2.3: Create symbol normalization and lookup rules

**Description**

Define how symbols are mapped between frontend input, backend canonical
representation, market-data providers, and brokers.

**Checklist**

- [ ] Define uppercase canonical ticker format.
- [ ] Keep exchange, asset type, and currency as separate normalized fields.
- [ ] Define provider-specific symbol translation rules.
- [ ] Handle unsupported or ambiguous symbols.
- [ ] Add lookup validation behavior.
- [ ] Document edge cases and assumptions.

### [ ] Ticket 2.4: Add backend market-data service layer

**Description**

Create backend services that call market-data adapters through internal
contracts and return normalized responses to the frontend.

**Checklist**

- [ ] Add service for latest quote lookup.
- [ ] Add service for historical bar requests.
- [ ] Add provider-status service.
- [ ] Add provider selection and fallback rules.
- [ ] Return normalized response models.

### [ ] Ticket 2.5: Add market-data API endpoints

**Description**

Expose backend endpoints for latest quotes, historical bars, and provider health
so the frontend can start integrating against stable APIs.

**Checklist**

- [ ] Add endpoint for latest quote lookup.
- [ ] Add endpoint for historical bars.
- [ ] Add endpoint for provider status.
- [ ] Validate symbol and interval inputs.
- [ ] Return actionable errors for unsupported requests.

### [ ] Ticket 2.6: Build frontend market-data validation screen

**Description**

Create a simple frontend screen that lets an operator enter a symbol, choose a
provider, view quote and bar data, and inspect provider status.

**Checklist**

- [ ] Add symbol input and provider selector.
- [ ] Display latest quote details.
- [ ] Display historical bars or chart data.
- [ ] Show loading, error, and stale-data states.
- [ ] Show provider connectivity status.

### [ ] Ticket 2.7: Add provider failover rules

**Description**

Define and implement how the backend falls back between Massive and BarChart
when data is missing, stale, or unavailable.

**Checklist**

- [ ] Define the primary provider selection rules.
- [ ] Define fallback trigger conditions.
- [ ] Define how the chosen provider is surfaced to the UI.
- [ ] Log failover events.
- [ ] Document expected behavior for unsupported symbols.

## Epic 3: Broker Connectivity Baseline

**Description**

Connect brokers for account visibility and controlled order workflows, starting
with Interactive Brokers and then adding TradeZero.

**Epic Checklist**

- [ ] Interactive Brokers works end to end for visibility and test orders.
- [ ] TradeZero works end to end for visibility and test orders.
- [ ] Backend exposes normalized broker APIs.
- [ ] Frontend supports account and order monitoring.
- [ ] Order states are normalized across brokers.

### [ ] Ticket 3.1: Implement Interactive Brokers adapter

**Description**

Build the Interactive Brokers adapter in `aptrade` for connection health,
account summary, positions, open orders, place order, cancel order, and order
status updates.

**Checklist**

- [ ] Add Interactive Brokers configuration and credentials handling.
- [ ] Implement connection-health reporting.
- [ ] Implement account summary retrieval.
- [ ] Implement positions and open orders retrieval.
- [ ] Implement place and cancel order support.

### [ ] Ticket 3.2: Implement TradeZero adapter

**Description**

Build the TradeZero adapter in `aptrade` with the same baseline feature set as
Interactive Brokers, while capturing broker-specific constraints.

**Checklist**

- [ ] Add TradeZero configuration and credentials handling.
- [ ] Implement connection-health reporting.
- [ ] Implement account summary retrieval.
- [ ] Implement positions and open orders retrieval.
- [ ] Implement place and cancel order support.

### [ ] Ticket 3.3: Normalize broker order-state machine

**Description**

Define the internal order-state model and map each broker's native order
lifecycle into that shared state machine.

**Checklist**

- [ ] Define canonical order states.
- [ ] Map Interactive Brokers statuses to canonical states.
- [ ] Map TradeZero statuses to canonical states.
- [ ] Define handling for unknown or partial states.
- [ ] Document state-transition rules.

### [ ] Ticket 3.4: Add backend broker service layer

**Description**

Create backend services for broker connection status, account selection, account
summary retrieval, positions, open orders, and order actions.

**Checklist**

- [ ] Add service for broker connection status.
- [ ] Add service for account summaries.
- [ ] Add service for positions and open orders.
- [ ] Add service for order placement and cancellation.
- [ ] Add broker selection and routing rules.

### [ ] Ticket 3.5: Add backend broker API endpoints

**Description**

Expose normalized broker APIs to the frontend for account visibility and
controlled test-order workflows.

**Checklist**

- [ ] Add endpoint for broker status.
- [ ] Add endpoint for account summary.
- [ ] Add endpoint for positions and open orders.
- [ ] Add endpoint for place order.
- [ ] Add endpoint for cancel order.

### [ ] Ticket 3.6: Build frontend broker operations screen

**Description**

Create a frontend screen for broker connection status, account summaries,
positions, open orders, and manual test-order submission behind a safety
control.

**Checklist**

- [ ] Add broker selector and account selector.
- [ ] Display connection and health status.
- [ ] Display account balances, positions, and open orders.
- [ ] Add manual test-order form.
- [ ] Add confirmation and error handling for order actions.

### [ ] Ticket 3.7: Add live-trading safety flag and permissions gate

**Description**

Introduce explicit feature flags and permission checks that restrict manual and
automated live order placement until the system is ready.

**Checklist**

- [ ] Add environment or configuration flag for live trading.
- [ ] Add paper-versus-live guardrails.
- [ ] Restrict manual live order entry to authorized users.
- [ ] Show current trading mode in the UI.
- [ ] Log blocked actions for auditability.

## Epic 4: Strategy Runtime and Paper Trading

**Description**

Connect market data, execution, and operator controls so strategies can run in
paper mode with full visibility and audit trails.

**Epic Checklist**

- [ ] Strategy lifecycle is defined.
- [ ] Strategy inputs are configurable.
- [ ] Paper-trading flow works end to end.
- [ ] Strategy actions are visible in the UI.
- [ ] Audit logs exist for automated actions.

### [ ] Ticket 4.1: Define strategy lifecycle and state model

**Description**

Define the lifecycle states and transitions for strategies, including draft,
paper, live, paused, and stopped.

**Checklist**

- [ ] Define valid lifecycle states.
- [ ] Define allowed transitions.
- [ ] Define failure and recovery states.
- [ ] Document operator actions that trigger transitions.
- [ ] Document guardrails for live activation.

### [ ] Ticket 4.2: Define strategy configuration contract

**Description**

Create the normalized configuration model for strategies, including symbols,
timeframe, broker, provider, sizing, and risk parameters.

**Checklist**

- [ ] Define required strategy fields.
- [ ] Define optional strategy fields.
- [ ] Define validation rules.
- [ ] Define default values where appropriate.
- [ ] Document how strategy config maps to runtime behavior.

### [ ] Ticket 4.3: Implement backend strategy orchestration service

**Description**

Build the backend orchestration path that accepts signals, runs pre-trade
checks, generates orders, reconciles fills, and updates positions.

**Checklist**

- [ ] Accept and validate strategy signals.
- [ ] Run pre-trade validations.
- [ ] Generate normalized order requests.
- [ ] Reconcile fills and update state.
- [ ] Emit strategy and execution events.

### [ ] Ticket 4.4: Add paper-trading execution path

**Description**

Add a paper-trading mode that exercises the strategy lifecycle and execution
flow without sending live broker orders.

**Checklist**

- [ ] Define paper execution behavior.
- [ ] Simulate or route paper orders safely.
- [ ] Record paper fills and position changes.
- [ ] Keep paper and live state clearly separated.
- [ ] Surface paper mode status in the UI.

### [ ] Ticket 4.5: Build frontend strategy management screens

**Description**

Create frontend pages to create, edit, activate, pause, and inspect strategies,
including the broker and provider assigned to each strategy.

**Checklist**

- [ ] Add strategy list view.
- [ ] Add create and edit strategy workflow.
- [ ] Add broker and provider assignment controls.
- [ ] Add strategy status and recent activity view.
- [ ] Add pause and stop controls.

### [ ] Ticket 4.6: Add strategy audit trail and event log

**Description**

Provide a full history of automated actions, state changes, signals, generated
orders, and operator interventions.

**Checklist**

- [ ] Define audit event types.
- [ ] Persist strategy action history.
- [ ] Persist operator intervention history.
- [ ] Display recent strategy events in the UI.
- [ ] Ensure events can be correlated to orders and fills.

## Epic 5: Risk, Reliability, and Controls

**Description**

Add the safeguards, health monitoring, and failure handling needed to run the
system safely and diagnose problems quickly.

**Epic Checklist**

- [ ] Pre-trade risk checks exist.
- [ ] Circuit breakers and emergency stop exist.
- [ ] Broker and provider heartbeats exist.
- [ ] Reconnection and stale-state handling exist.
- [ ] Critical alerting is defined.

### [ ] Ticket 5.1: Implement pre-trade risk engine

**Description**

Create the risk checks that run before any order is submitted, including size
limits, trading windows, symbol allowlists, daily loss thresholds, and duplicate
order prevention.

**Checklist**

- [ ] Add max position size checks.
- [ ] Add max daily loss checks.
- [ ] Add allowed-symbol checks.
- [ ] Add trading-window checks.
- [ ] Add duplicate-order prevention.

### [ ] Ticket 5.2: Add emergency stop and circuit breakers

**Description**

Implement mechanisms that can immediately stop automated trading or block order
flow when critical conditions are detected.

**Checklist**

- [ ] Define emergency-stop behavior.
- [ ] Define circuit-breaker trigger conditions.
- [ ] Block new automated orders when engaged.
- [ ] Surface current protection state in the UI.
- [ ] Audit all activations and releases.

### [ ] Ticket 5.3: Add health checks and heartbeats

**Description**

Create recurring health and heartbeat checks for brokers, providers, and key
internal services.

**Checklist**

- [ ] Define health-check cadence.
- [ ] Add broker heartbeat monitoring.
- [ ] Add data-provider heartbeat monitoring.
- [ ] Add internal service health summary.
- [ ] Surface degraded health in the UI.

### [ ] Ticket 5.4: Implement reconnect and stale-state handling

**Description**

Handle disconnects, stale quotes, stale order state, and partial recovery so the
system fails safely instead of drifting into an unknown state.

**Checklist**

- [ ] Define reconnect policies per provider and broker.
- [ ] Detect stale quote conditions.
- [ ] Detect stale order or position state.
- [ ] Define safe behavior during degraded connectivity.
- [ ] Log recovery attempts and outcomes.

### [ ] Ticket 5.5: Add alerting for critical failures

**Description**

Define and implement alerting for disconnections, rejected orders, repeated
reconnect failures, and other critical execution or data issues.

**Checklist**

- [ ] Define alert severity levels.
- [ ] Alert on broker disconnects.
- [ ] Alert on provider disconnects or stale data.
- [ ] Alert on rejected or unacknowledged orders.
- [ ] Document alert destinations and escalation path.

### [ ] Ticket 5.6: Add event replay and debugging support

**Description**

Provide tooling to inspect past order and market-data event sequences so
failures can be understood and reproduced.

**Checklist**

- [ ] Define retained event types.
- [ ] Define event replay scope and filters.
- [ ] Add event correlation support.
- [ ] Document replay workflow.
- [ ] Ensure sensitive data is handled appropriately.

## Epic 6: Production Readiness and Operations

**Description**

Prepare the system for repeatable deployment, support, testing, monitoring, and
credential management.

**Epic Checklist**

- [ ] Integration coverage exists for adapters and service flows.
- [ ] Deployment profiles are defined.
- [ ] Secrets are managed appropriately.
- [ ] Operational runbooks exist.
- [ ] Metrics and dashboards are defined.

### [ ] Ticket 6.1: Build integration test suite for adapters

**Description**

Create integration tests that verify market-data providers and brokers conform
to the internal contracts and handle common failure modes.

**Checklist**

- [ ] Add tests for Massive adapter.
- [ ] Add tests for BarChart adapter.
- [ ] Add tests for Interactive Brokers adapter.
- [ ] Add tests for TradeZero adapter.
- [ ] Add tests for contract error handling.

### [ ] Ticket 6.2: Add paper-trading regression flows

**Description**

Create repeatable regression scenarios that validate market-data ingestion,
strategy execution, order generation, and state updates in paper mode.

**Checklist**

- [ ] Define regression scenarios.
- [ ] Add one end-to-end paper strategy scenario.
- [ ] Add one broker-visibility scenario.
- [ ] Add one provider failover scenario.
- [ ] Document expected outputs.

### [ ] Ticket 6.3: Define deployment profiles

**Description**

Document and implement local, staging, and production deployment profiles with
environment-specific configuration rules.

**Checklist**

- [ ] Define local deployment profile.
- [ ] Define staging deployment profile.
- [ ] Define production deployment profile.
- [ ] Document environment-specific settings.
- [ ] Document release prerequisites.

### [ ] Ticket 6.4: Implement secrets and credential management

**Description**

Define how broker credentials, provider credentials, and sensitive environment
settings are stored, rotated, and accessed.

**Checklist**

- [ ] Identify all sensitive settings.
- [ ] Define secret-storage approach.
- [ ] Define credential rotation process.
- [ ] Prevent secrets from reaching logs.
- [ ] Document local-development handling.

### [ ] Ticket 6.5: Write operational runbooks

**Description**

Create runbooks for common operational events, including disconnects, failed
reconnects, rejected orders, stale data, and partial outages.

**Checklist**

- [ ] Write broker disconnect runbook.
- [ ] Write provider outage runbook.
- [ ] Write rejected-order runbook.
- [ ] Write stale-data runbook.
- [ ] Write safe-restart runbook.

### [ ] Ticket 6.6: Define metrics and dashboard requirements

**Description**

Identify the metrics needed to operate the platform, then define a dashboard
that highlights latency, uptime, fill rates, rejection rates, and stale-state
signals.

**Checklist**

- [ ] Define broker latency metrics.
- [ ] Define provider latency metrics.
- [ ] Define order rejection and fill metrics.
- [ ] Define stale-data and stale-state metrics.
- [ ] Define dashboard sections and owners.

## Epic 7: Scheduling and Background Jobs

**Description**

Add the execution infrastructure for daily pipelines, on-demand background work,
and short-interval recurring jobs using Airflow, Celery workers, and Celery
Beat.

**Epic Checklist**

- [ ] Airflow handles daily and pipeline-style jobs.
- [ ] Celery workers handle on-demand jobs.
- [ ] Celery Beat handles recurring short-interval jobs.
- [ ] Backend exposes task submission and status APIs.
- [ ] Task observability exists across all execution paths.

### [ ] Ticket 7.1: Define task ownership by execution system

**Description**

Create the rules that determine which jobs run in Airflow, which run in Celery
workers, and which run in Celery Beat so the system has a stable scheduling
model.

**Checklist**

- [ ] Confirm daily pipeline jobs that belong in Airflow.
- [ ] Confirm on-demand jobs that belong in Celery workers.
- [ ] Confirm short recurring jobs that belong in Celery Beat.
- [ ] Define jobs that must not run in Airflow.
- [ ] Document the decision rules for future tasks.

### [ ] Ticket 7.2: Add Airflow infrastructure for daily pipelines

**Description**

Set up Airflow as the orchestration layer for daily long-running pipelines such
as data downloads, cleaning, enrichment, reconciliation, and scheduled
reporting.

**Checklist**

- [ ] Define how Airflow is deployed in local and non-local environments.
- [ ] Define connection and secret handling for Airflow jobs.
- [ ] Define DAG folder structure and ownership.
- [ ] Add one initial DAG for the daily data pipeline.
- [ ] Record DAG run status and failures in a way the platform can inspect.

### [ ] Ticket 7.3: Build daily data pipeline DAG in Airflow

**Description**

Implement the first production-relevant Airflow DAG for daily download,
processing, cleaning, and post-run validation of market data.

**GitHub Metadata**

- Labels: `epic:scheduling`, `area:airflow`, `area:backend`, `type:pipeline`,
  `priority:p0`
- Milestone: `M4 Scheduling Core`
- Depends on: `1.1`, `1.3`, `1.4`, `7.1`, `7.2`, `2.3`

**Technical Scope**

- Create a DAG named `daily_market_data_pipeline`.
- Schedule it once per trading day with explicit timezone handling.
- Split the DAG into discrete tasks for download, process, clean, validate, and
  publish completion status.
- Use idempotent task design so reruns do not corrupt stored datasets.
- Store run metadata including start time, end time, duration, status, row
  counts, and validation summary.
- Emit a normalized run result the backend can query without depending on the
  Airflow UI.

**Acceptance Criteria**

- The DAG can be triggered manually and by schedule.
- A failed task can be retried without duplicating output artifacts.
- Validation failures mark the DAG run as failed with a clear reason.
- The backend can read the last run status and timestamps.
- The DAG has task-level logs and owner metadata.

**Checklist**

- [ ] Add download step.
- [ ] Add processing step.
- [ ] Add cleaning and validation step.
- [ ] Add retry and failure handling.
- [ ] Add success and failure notifications.
- [ ] Set DAG id, schedule, timezone, retries, and owner metadata.
- [ ] Persist run metadata for backend status lookup.
- [ ] Make task outputs idempotent for reruns and backfills.

### [ ] Ticket 7.4: Add Celery worker infrastructure for on-demand jobs

**Description**

Set up Celery and the message broker so the backend can dispatch background jobs
such as backtests, analysis runs, ad hoc recomputations, and user-triggered
refreshes.

**Checklist**

- [ ] Choose and configure the Celery broker and result backend.
- [ ] Define worker process layout and queues.
- [ ] Define task registration and discovery conventions.
- [ ] Add backend integration for task submission.
- [ ] Add worker health and queue visibility.

### [ ] Ticket 7.5: Implement backtest job as first Celery task

**Description**

Use backtesting as the first on-demand Celery job so the system proves task
submission, progress tracking, completion, and error handling for user-triggered
background work.

**GitHub Metadata**

- Labels: `epic:scheduling`, `area:celery`, `area:backend`, `type:worker`,
  `priority:p0`
- Milestone: `M4 Scheduling Core`
- Depends on: `1.1`, `7.1`, `7.4`, `7.8`

**Technical Scope**

- Define a Celery task named `run_backtest` with a versioned input payload.
- Inputs should include strategy identifier or configuration snapshot, symbol
  set, date range, timeframe, and execution mode.
- Persist task state transitions such as queued, running, succeeded, failed, and
  cancelled if supported.
- Expose progress updates from the worker so the backend can report intermediate
  status.
- Persist result metadata separately from large result artifacts.
- Failures should return structured error codes for invalid input, dependency
  failure, timeout, and internal error.

**Acceptance Criteria**

- The backend can enqueue a backtest and receive a task identifier.
- The worker updates progress during execution.
- The backend can query the final task result and summarized metrics.
- Invalid requests fail fast before expensive execution.
- Failed tasks record actionable error information.

**Checklist**

- [ ] Define backtest input contract.
- [ ] Submit backtest from backend to Celery.
- [ ] Track status and progress.
- [ ] Persist result metadata.
- [ ] Return actionable errors on failure.
- [ ] Define task state model and result summary schema.
- [ ] Add timeout and retry rules appropriate for long-running jobs.
- [ ] Ensure large result artifacts are stored outside transient task state.

### [ ] Ticket 7.6: Add Celery Beat for minute-level recurring jobs

**Description**

Set up Celery Beat for frequent recurring tasks such as every-minute polling,
heartbeat checks, and periodic synchronization jobs.

**Checklist**

- [ ] Define Beat schedule storage and configuration.
- [ ] Add one-minute polling schedule.
- [ ] Add heartbeat or health-check schedule.
- [ ] Prevent overlapping runs where required.
- [ ] Record last-run and next-run metadata.

### [ ] Ticket 7.7: Implement minute polling task

**Description**

Build the first short-interval recurring task that polls data every minute,
normalizes results, and records health or freshness status.

**GitHub Metadata**

- Labels: `epic:scheduling`, `area:celery-beat`, `area:aptrade`, `area:backend`,
  `type:worker`, `priority:p0`
- Milestone: `M4 Scheduling Core`
- Depends on: `1.1`, `1.3`, `2.3`, `7.1`, `7.6`

**Technical Scope**

- Implement one minute-level polling task for a clearly defined target such as
  quotes, provider heartbeat, or symbol freshness checks.
- Use a lock or equivalent overlap prevention so a delayed run does not create
  concurrent duplicate polls.
- Normalize the fetched payload into internal quote or health models before
  persistence.
- Record freshness timestamps, source status, retry count, and last error.
- Respect provider rate limits and backoff rules.
- Emit metrics the backend can use to show stale or degraded status.

**Acceptance Criteria**

- The Beat schedule triggers the task every minute.
- Overlapping runs are prevented or explicitly handled.
- The task stores a last-success timestamp and last-failure reason.
- Rate-limit responses are retried with backoff and do not spam the provider.
- The backend can determine whether the polled source is fresh or stale.

**Checklist**

- [ ] Define the polling target and interval.
- [ ] Fetch and normalize the polled data.
- [ ] Handle rate limits and retries.
- [ ] Mark stale or failed polls.
- [ ] Surface freshness status to the backend.
- [ ] Prevent overlapping execution for delayed runs.
- [ ] Record last-success, last-failure, and freshness timestamps.
- [ ] Emit health metrics for stale or degraded polling.

### [ ] Ticket 7.8: Add backend task submission and status APIs

**Description**

Expose backend APIs that let the frontend submit on-demand jobs, query task
status, and inspect recent scheduler activity without depending on Airflow or
Celery internals.

**Checklist**

- [ ] Add endpoint for submitting supported on-demand tasks.
- [ ] Add endpoint for task status lookup.
- [ ] Add endpoint for recent task history.
- [ ] Normalize status and error responses.
- [ ] Enforce permissions and audit logging.

### [ ] Ticket 7.9: Build frontend task operations screen

**Description**

Create a frontend screen for launching supported on-demand tasks, checking
progress, and reviewing recent scheduled-task outcomes.

**Checklist**

- [ ] Add on-demand task submission controls.
- [ ] Display running, completed, and failed tasks.
- [ ] Display progress and result summaries.
- [ ] Display scheduler-specific errors in normalized form.
- [ ] Show last successful daily and minute-level runs.

### [ ] Ticket 7.10: Add observability for schedulers and workers

**Description**

Track execution duration, queue depth, retries, failures, and last-success
timestamps across Airflow, Celery workers, and Celery Beat.

**Checklist**

- [ ] Define metrics for Airflow DAG runs.
- [ ] Define metrics for Celery queues and workers.
- [ ] Define metrics for recurring Beat tasks.
- [ ] Add failure and retry visibility.
- [ ] Add last-success timestamps per job.

## Suggested First Batch of GitHub Issues

If you want the minimum set to start implementation, create these first:

- [ ] Ticket 1.1: Define canonical trading domain models
- [ ] Ticket 1.2: Define broker adapter interface
- [ ] Ticket 1.3: Define market-data adapter interface
- [ ] Ticket 1.4: Create provider and broker capability matrix
- [ ] Ticket 2.1: Implement Massive market-data adapter
- [ ] Ticket 2.4: Add backend market-data service layer
- [ ] Ticket 2.5: Add market-data API endpoints
- [ ] Ticket 2.6: Build frontend market-data validation screen
- [ ] Ticket 3.1: Implement Interactive Brokers adapter
- [ ] Ticket 3.4: Add backend broker service layer
- [ ] Ticket 3.5: Add backend broker API endpoints
- [ ] Ticket 3.6: Build frontend broker operations screen
- [ ] Ticket 7.1: Define task ownership by execution system
- [ ] Ticket 7.2: Add Airflow infrastructure for daily pipelines
- [ ] Ticket 7.4: Add Celery worker infrastructure for on-demand jobs
- [ ] Ticket 7.6: Add Celery Beat for minute-level recurring jobs

## GitHub Import Order

Create issues in this order so dependencies are already present in GitHub when
later tickets are opened.

1. Epic issues: `Epic 1` through `Epic 7`
2. Foundation tickets: `1.1` through `1.6`
3. Scheduling foundation tickets: `7.1`, `7.2`, `7.4`, `7.6`, `7.8`
4. Data baseline tickets: `2.3`, `2.1`, `2.4`, `2.5`, `2.6`
5. First scheduler implementation tickets: `7.3`, `7.7`, `7.5`, `7.9`, `7.10`
6. Broker baseline tickets: `3.1`, `3.3`, `3.4`, `3.5`, `3.6`, `3.7`
7. Secondary integrations: `2.7`, `2.2`, `3.2`
8. Strategy runtime tickets: `4.1` through `4.6`
9. Risk and reliability tickets: `5.1` through `5.6`
10. Operations tickets: `6.1` through `6.6`

## Confirmed Working Defaults

These are the current defaults. Only revisit them if implementation or operating
constraints prove they are wrong:

- [ ] Massive is the default real-time provider; BarChart is the fallback and
      secondary source.
- [ ] Interactive Brokers is the first live-trading target after paper mode.
- [ ] Live strategy orchestration stays in the backend; Celery workers are for
      on-demand background jobs.
- [ ] `aptrade` remains an internal Python package in the first release.
- [ ] Canonical symbols use an uppercase ticker, with exchange, asset type, and
      currency stored separately.
- [ ] The first Airflow DAG covers daily market-data download, normalize, clean,
      validate, and report.
- [ ] The first on-demand Celery task is backtesting, followed by analysis and
      ad hoc refresh jobs.
- [ ] The first Celery Beat jobs are provider-health checks and minute-level
      data freshness polling.
