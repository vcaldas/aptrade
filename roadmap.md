# Trading System Roadmap

## Purpose

Build a trading system with a clear separation of responsibilities:

- `frontend` owns the trader-facing UI and workflows in TypeScript.
- `backend` owns orchestration, business rules, APIs, persistence, and automation in Python.
- `aptrade` owns lower-level broker and market-data actions, adapters, and execution primitives.

This first roadmap focuses on:

- Broker support for Interactive Brokers and TradeZero.
- Market data support for Massive and BarChart.
- A system design that allows new brokers and data providers to be added without rewriting the UI or core backend flows.

## Product Goals

- View live and historical market data from supported providers.
- Manage accounts, balances, positions, orders, and executions across multiple brokers.
- Run strategy workflows from a unified interface.
- Keep broker-specific and provider-specific logic isolated behind stable internal interfaces.
- Make it possible to test strategies and broker flows before enabling live trading.

## Codebase Responsibilities

### Frontend

- Strategy configuration and monitoring screens.
- Broker and data-provider connection status.
- Account, positions, orders, fills, and logs views.
- Operator controls for enabling paper trading, live trading, and emergency stop.
- Alerts and error visibility for disconnected brokers or stale data streams.

### Backend

- Public API consumed by the frontend.
- Domain models for accounts, instruments, orders, executions, positions, and strategies.
- Authentication, authorization, and audit logging.
- Strategy orchestration, task submission, and task status APIs.
- Persistence for configuration, market snapshots, order state, fills, and system events.
- Risk checks, broker routing rules, and execution policies.

### Scheduling and Workers

- Airflow owns long-running daily and batch pipelines.
- Celery workers own on-demand background jobs triggered by users or internal services.
- Celery Beat owns short-interval recurring jobs such as minute polling and health checks.
- The backend remains the control plane for submitting jobs, checking status, and enforcing permissions.

### aptrade

- Broker adapters for Interactive Brokers and TradeZero.
- Market-data adapters for Massive and BarChart.
- Unified low-level interfaces for:
  - market data subscription and polling
  - historical data retrieval
  - order placement, modification, and cancellation
  - account and position retrieval
  - execution and order-status events
- Retry, reconnect, rate-limit handling, and normalization of vendor-specific payloads.

## Guiding Architecture

### 1. Adapter-first design

Define internal interfaces before finishing integrations:

- `BrokerAdapter`
- `MarketDataAdapter`
- `ExecutionEvent`
- `OrderRequest`
- `OrderStatus`
- `PositionSnapshot`
- `QuoteBarTick` or equivalent normalized market-data models

Each external integration should map into these internal types.

### 2. Separate orchestration from transport

The backend should decide what the system does.
`aptrade` should decide how to talk to each external broker or data vendor.

### 3. Support both polling and streaming

Not every provider will expose the same real-time behavior. The system should support:

- streaming subscriptions when available
- polling fallbacks when required
- consistent downstream events regardless of transport style

### 4. Build paper-first, then live

Every broker integration should be proven in:

- local mocks
- sandbox or paper environments where available
- controlled live rollout with broker-specific feature flags

### 5. Use the right scheduler for each task shape

- Airflow for daily and dependency-heavy pipelines such as downloads, cleaning, enrichment, and end-of-day processing.
- Celery workers for on-demand jobs such as backtests, analytics, ad hoc rebuilds, and user-triggered workflows.
- Celery Beat for short recurring jobs such as every-minute polling, heartbeats, and stale-state detection.
- Avoid using Airflow for minute-level polling or low-latency broker-facing execution paths.

## Phase Roadmap

## Phase 0: Foundation and Contracts

Objective: create stable interfaces so integrations do not leak vendor behavior into the rest of the system.

Deliverables:

- Define canonical models for instruments, quotes, bars, orders, fills, positions, balances, and strategy signals.
- Define adapter interfaces in `aptrade` for brokers and market data.
- Define backend services that depend only on internal adapter contracts.
- Add environment-based provider and broker configuration.
- Add a capability matrix for each integration, such as:
  - live quotes
  - historical bars
  - order types
  - premarket and after-hours support
  - paper trading support
  - short locate or margin-specific behaviors if applicable
- Add structured logging and correlation IDs across frontend, backend, and `aptrade`.

Exit criteria:

- New providers can plug in without changing frontend screens.
- Backend orchestration paths are independent of vendor payload formats.

## Phase 1: Market Data Baseline

Objective: deliver normalized market data first, because strategies and execution logic depend on it.

Priority order:

1. Massive
2. BarChart

Deliverables:

- Implement `MassiveMarketDataAdapter` in `aptrade`.
- Implement `BarChartMarketDataAdapter` in `aptrade`.
- Normalize symbols, timestamps, sessions, and interval handling.
- Support at minimum:
  - latest quote retrieval
  - historical bar retrieval
  - provider health status
- Expose backend endpoints for:
  - provider status
  - historical chart requests
  - latest quote lookup
- Add frontend screens or panels for:
  - provider connectivity
  - symbol lookup validation
  - quote and bar visualization

Default decisions:

- Massive is the default source for real-time quotes in the first release.
- BarChart is the secondary and fallback provider, with emphasis on coverage and gap handling.
- Canonical instrument identity should separate `symbol`, `asset_type`, `primary_exchange`, and `currency` instead of encoding provider-specific formats into one string.

Exit criteria:

- A frontend user can choose a symbol and see normalized data regardless of provider.
- Backend can switch providers by configuration or failover rules.

## Phase 2: Broker Connectivity Baseline

Objective: establish account and order lifecycle connectivity before strategy automation.

Priority order:

1. Interactive Brokers
2. TradeZero

Deliverables:

- Implement `InteractiveBrokersAdapter` in `aptrade`.
- Implement `TradeZeroAdapter` in `aptrade`.
- Support at minimum:
  - connection health
  - account summary
  - positions
  - open orders
  - place order
  - cancel order
  - order status updates
- Normalize broker-specific order states into a shared internal state machine.
- Add backend services for broker routing and account selection.
- Add frontend views for:
  - broker connection status
  - account summary
  - positions
  - order blotter
  - manual order entry for controlled testing

Default decisions:

- Each strategy should bind to one broker account in the first release.
- Live trading must remain behind explicit feature flags and permission checks.
- Interactive Brokers is the first live-trading target; TradeZero follows after the broker abstractions are proven.

Exit criteria:

- A user can connect to each broker, inspect account state, and submit controlled test orders.

## Phase 3: Strategy Runtime

Objective: connect data, execution, and operator workflows into a usable trading loop.

Deliverables:

- Define strategy lifecycle states: draft, paper, live, paused, stopped.
- Define strategy input contracts:
  - symbols
  - timeframe
  - data provider
  - broker
  - sizing rules
  - risk rules
- Add backend orchestration for:
  - signal ingestion
  - pre-trade checks
  - order creation
  - fill reconciliation
  - position updates
- Add frontend strategy pages for:
  - create and edit strategy
  - assign broker and data provider
  - inspect recent signals and actions
  - pause or stop execution
- Add audit trails for all automated actions.

Exit criteria:

- A configured strategy can run end to end in paper mode using supported data and broker integrations.

Default decisions:

- Live strategy orchestration should stay in the backend control plane for the first release.
- Celery workers should be used for on-demand background jobs such as backtests and heavy analysis, not as the primary live-trading runtime.

## Phase 4: Risk, Reliability, and Controls

Objective: make the system safe enough for controlled live use.

Deliverables:

- Pre-trade risk engine with checks for:
  - max position size
  - max daily loss
  - allowed symbols
  - allowed trading windows
  - duplicate order prevention
- Circuit breakers and emergency stop.
- Heartbeats for brokers and data providers.
- Automatic reconnection and stale-data detection.
- Alerting for disconnects, rejected orders, and unacknowledged state transitions.
- Replay tooling for order and market-data event debugging.

Exit criteria:

- The system can detect degraded states and fail safely.

## Phase 5: Production Readiness

Objective: move from functional to operable.

Deliverables:

- Integration test suite covering provider and broker adapters.
- Paper-trading regression flows.
- Deployment profiles for local, staging, and production.
- Secrets management for broker and data-provider credentials.
- Runbooks for outages, reconnects, and broker-specific failures.
- Metrics dashboard for latency, fill rates, rejected orders, stale quotes, and provider uptime.

Exit criteria:

- The team can deploy, monitor, and support the system without relying on tribal knowledge.

## Phase 6: Scheduling and Background Execution

Objective: establish a clear execution model for daily pipelines, on-demand jobs, and short-interval recurring tasks.

Deliverables:

- Add Airflow for daily pipelines such as data download, cleaning, enrichment, reconciliation, and scheduled reports.
- Add Celery workers for on-demand jobs such as backtests, analysis, ad hoc refreshes, and batch recomputations.
- Add Celery Beat for minute-level polling, heartbeat checks, and periodic synchronization.
- Define backend APIs for task submission, status retrieval, cancellation where supported, and audit logging.
- Define task ownership boundaries so trading runtime logic does not depend on Airflow DAG execution.
- Add observability for scheduled and background jobs, including duration, retries, queue depth, failures, and last-success timestamps.

Exit criteria:

- Daily pipelines run through Airflow.
- On-demand jobs run through Celery workers.
- Minute-level recurring jobs run through Celery Beat.
- The frontend can inspect task status without needing to know the underlying scheduler.

Default decisions:

- The first Airflow DAG should run daily market-data download, normalization, cleaning, validation, and completion reporting.
- The first Celery task should be a user-triggered backtest.
- The first Celery Beat jobs should be provider-health checks and minute-level data freshness polling.

## Initial Priority Stack

If the goal is fastest useful progress, build in this order:

1. Internal contracts and canonical models.
2. Scheduler and worker foundations for Airflow, Celery, and Celery Beat.
3. Symbol normalization and provider capability definitions.
4. Massive market-data adapter and backend data APIs.
5. Frontend market-data validation screens.
6. First Airflow DAG, first Celery task, and first Celery Beat recurring job.
7. Interactive Brokers adapter and broker APIs.
8. Frontend broker operations screens and live-trading safety controls.
9. BarChart as fallback data source.
10. TradeZero adapter.
11. Strategy runtime and paper-trading workflow.
12. Risk controls and production hardening.

## Near-Term Milestones

### Milestone 1: Foundations and Contracts

- Canonical models are defined for market data, orders, positions, executions, and strategy inputs.
- Broker and market-data adapter interfaces are defined.
- Logging, tracing, and capability-matrix conventions are documented.
- Core architecture defaults are recorded so implementation can proceed without reopening baseline decisions.

### Milestone 2: Scheduler and Worker Foundations

- Airflow infrastructure exists for daily pipelines.
- Celery worker infrastructure exists for on-demand jobs.
- Celery Beat exists for short recurring jobs.
- Backend task submission and task status APIs are available.

### Milestone 3: Data Visibility

- One symbol search flow in the frontend.
- Backend endpoint returning normalized quotes and bars.
- Massive connected end to end.

### Milestone 4: Scheduler Baseline

- Airflow runs the daily data pipeline.
- Celery runs one on-demand backtest job.
- Celery Beat runs one every-minute polling job with status tracking.

### Milestone 5: Broker Visibility

- Interactive Brokers account summary, positions, and open orders visible in the UI.
- Manual test order path available behind a safety flag.

### Milestone 6: Multi-provider and Multi-broker Baseline

- BarChart available as alternate data source.
- TradeZero connected for account state and test orders.
- Provider and broker capability matrix documented and surfaced in admin UI.

### Milestone 7: Paper Strategy Execution

- One strategy type runs in paper mode with full event logging.

## Current Defaults

- Use PostgreSQL as the primary database, with append-only event tables for orders, executions, task runs, and system events.
- Run daily and dependency-heavy pipelines in Airflow.
- Run on-demand background jobs in Celery workers.
- Run minute-level recurring jobs in Celery Beat.
- Keep live strategy orchestration in the backend control plane for the first release.
- Keep `aptrade` as an internal Python package first, and only move to a service boundary if scale or isolation requirements justify it.
- Use uppercase canonical symbols for the primary ticker and store exchange, asset type, and currency as separate fields.
- Use WebSockets for live operational updates where needed, with polling fallback for simple screens and degraded environments.

## Risks to Address Early

- Broker APIs may differ materially in order support and status semantics.
- Market-data entitlements and rate limits may constrain features.
- Symbol identity mismatches can create execution risk.
- Reconnect logic and stale-state handling are often harder than initial happy-path integration.
- Live trading needs stronger controls than paper trading from the start of the design.

## Suggested First Sprint

- Finalize adapter interfaces and normalized domain models.
- Set up the first scheduler foundations for Airflow, Celery, and Celery Beat.
- Implement provider and broker capability definitions.
- Deliver Massive historical and latest-quote support through backend APIs.
- Build a frontend market-data test screen.
- Add Interactive Brokers connection health and account summary path.

## Definition of Success for This First Roadmap

You should be able to say the platform has a solid foundation when:

- the frontend does not know vendor-specific details
- the backend orchestrates against stable internal contracts
- `aptrade` contains vendor-specific integration code
- one data provider and one broker work end to end in a controlled flow
- daily pipelines, on-demand jobs, and minute-level recurring jobs each have a clear execution path
- adding the second provider and second broker is incremental, not architectural
