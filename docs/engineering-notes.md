---
title: Engineering notes
description: >-
  Deeper documents for contributors and for people evaluating how fasterRag is built: design,
  reliability, evidence, and project process. None of them is needed to use fasterRag.
---

# Engineering notes

You do not need any of these to use fasterRag. They are for contributors, and for anyone who wants
to check how the project is built before depending on it.

## Design

| Document | What it covers |
| --- | --- |
| [Repository structure](structure.md) | The layout of the repository and what each layer is responsible for |
| [End-to-end flows](flow.md) | The main flows through the system, as diagrams |
| [Internals](internals.md) | Which module implements each specified behaviour, and the test that proves it |
| [Data model](data-model.md) | The canonical entities and their invariants |
| [Prompt contracts](prompts.md) | The four places an LLM is called, and exactly what is sent |
| [Portability archive](archive-format.md) | The export and import file format |
| [Decision records](adr/ADR-0001-qdrant-as-reference-vector-db.md) | Why the main design choices were made |
| [Glossary](glossary.md) | The project's terms, with one meaning each |

## Reliability

| Document | What it covers |
| --- | --- |
| [Reliability](reliability.md) | The error taxonomy and the resilience patterns |
| [Failure modes](failure-modes.md) | What can fail, what happens, and the test for each |
| [Service levels](slo.md) | Indicators, targets and the error-budget policy |
| [Disaster recovery](disaster-recovery.md) | Backups, the restore drill, recovery objectives |

## Evidence

| Document | What it covers |
| --- | --- |
| [Benchmark ledger](benchmarks.md) | Every measurement, with method, hardware and commit |
| [Performance methodology](performance.md) | How performance is measured and what the goals are |
| [Testing strategy](testing-strategy.md) | The test tiers and the CI gates |
| [Verification procedure](verification.md) | How to prove a checkout matches the documents |
| [Release readiness](release-readiness.md) | The confidence assessment behind a release decision |

## Project

| Document | What it covers |
| --- | --- |
| [Scope and goals](scope.md) | What the project is for, and what it will not do |
| [Differentiators](differentiators.md) | The flagship capabilities and the status of each |
| [Release process](release.md) | How a release is made and what authorises it |
| [Rebuild playbook](rebuild-playbook.md) | The order the system was built in, and the traps found |
| [References](references.md) | External sources behind sourced claims |
