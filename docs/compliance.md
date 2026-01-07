# Curriculum Compliance Log

This document tracks how each curriculum requirement was satisfied, modified, or intentionally deviated from.

The goal is **explicit accountability**, not blind adherence.

---

## Module 1 — Infrastructure Archaeology

**Requirement:** Infrastructure as Code  
**Implementation:** Terraform-based VM provisioning  
**Status:** Compliant  

**Requirement:** Environment as Code  
**Implementation:** Docker for training and inference  
**Status:** Compliant  

**Notes:**  
Manual configuration was explicitly avoided. Any manual fix was rolled back and codified.

---

## Module 2 — Data Backbone

**Requirement:** Automated ETL  
**Implementation:** Scheduled DAG with idempotent transforms  
**Status:** Compliant  

**Requirement:** Feature Store  
**Implementation:** Centralized feature definitions  
**Status:** Compliant  

**Deviation:**  
Reservoir sampling not implemented due to fixed-rate synthetic stream.  
**Risk Introduced:** Potential bias under burst traffic.

---

## Module 3 — Model Debt Defense

**Requirement:** Time-aware splits  
**Implementation:** Temporal partitioning  
**Status:** Compliant  

**Requirement:** Leakage detection  
**Implementation:** SHAP analysis  
**Status:** Compliant  

---

## Module 4 — Deployment & Serving

**Requirement:** Blue-Green deployments  
**Implementation:** Dual environments with traffic switch  
**Status:** Compliant  

**Requirement:** Circuit breaker  
**Implementation:** Error-rate-based traffic routing  
**Status:** Compliant  

---

## Module 5 — Industry Masterclass

**Requirement:** Edge optimization  
**Implementation:** INT8 quantization  
**Status:** Compliant  

**Requirement:** Human-in-the-loop governance  
**Implementation:** Risk-based decision routing  
**Status:** Compliant  

---

## Summary

All core curriculum objectives were implemented or explicitly justified.  
Known risks and deviations are documented with mitigation strategies.
