# System Architecture

## High-Level Overview

The system is composed of loosely coupled subsystems designed to minimize cascading failures and enable independent evolution.

Raw Events
↓
ETL Pipeline (Batch / Streaming)
↓
Feature Store ←→ Training Pipeline
↓ ↓
Inference Service ← Model Registry
↓
Decision Routing (Automated vs Human)
↓
Monitoring & Feedback Loops


## Architectural Goals

- Prevent training–serving skew
- Enable safe model replacement
- Support rollback at every layer
- Ensure traceability from prediction to raw data
- Encode governance constraints directly in serving logic

---

## Core Components

### 1. Infrastructure Layer
- Provisioned via Terraform
- Immutable environments
- Containerized workloads

**Failure Mode Addressed:** Snowflake servers, environment drift

---

### 2. Data Layer
- Raw data is immutable
- ETL pipelines are idempotent
- Feature definitions are centralized

**Failure Mode Addressed:** Data leakage, hidden preprocessing differences

---

### 3. Feature Store
- Single source of truth for features
- Shared between training and inference

**Failure Mode Addressed:** Training–serving skew

---

### 4. Model Development
- Time-aware data splits
- Explicit imbalance handling
- Calibration and invariance testing

**Failure Mode Addressed:** Model debt, correction cascades

---

### 5. Serving Layer
- Stateless inference API
- Blue-Green deployments
- Circuit breaker protection

**Failure Mode Addressed:** Downtime, uncontrolled rollouts

---

### 6. Monitoring & Operations
- Latency and error SLAs
- Drift detection
- Performance degradation alerts

**Failure Mode Addressed:** Silent failures

---

### 7. Governance & Human-in-the-Loop
- Risk-based routing
- Human review for high-risk predictions
- Automated handling for low-risk predictions

**Failure Mode Addressed:** Unsafe automation

---

## Design Tradeoffs

- Chose robustness over peak accuracy
- Accepted higher latency to gain observability
- Introduced operational overhead to ensure traceability

All tradeoffs are documented in Architecture Decision Records (ADRs).

---

## Model Replaceability Test (Critical)

A core invariant of this architecture:

> Any trained model can be replaced with a worse or better one **without modifying infrastructure, pipelines, or serving interfaces**.

If this invariant is violated, the system is considered architecturally unsound.