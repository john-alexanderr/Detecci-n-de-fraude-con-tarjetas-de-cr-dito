# Business Requirements and Benchmark Scope

## Problem

The academic benchmark studies how transaction features rank possible fraud under severe class imbalance. In a real payment flow, a flagged transaction could be declined, sent for additional verification, or reviewed. Missing fraud can increase loss and dispute work; false alerts can create customer friction and lost revenue.

## Evaluation goals

- Compare binary classifiers using Average Precision, precision, recall, F2, and confusion-matrix counts.
- Make the false-alert burden visible alongside fraud detection.
- Keep each validation and test partition at the dataset's natural class prevalence.
- Prevent identical input-feature signatures from crossing the outer or inner split.
- Keep the outer test partition untouched until model choices are frozen.

## Supported claims

The current work supports descriptive analysis of a public historical dataset and one preliminary grouped validation comparison. It does not estimate institution-specific financial costs, identify a business-approved operating threshold, or establish a deployable fraud policy.

The dataset can support further academic experiments in schema and data-quality audits, feature-distribution analysis, rare-event classification, threshold trade-offs, and model-only inference-time benchmarking in a stated local environment. A chronological split could provide a limited historical simulation after verifying row ordering, but two days of data provide weak evidence about temporal generalization.

## Out of scope

The dataset has no card or customer identifiers, merchant attributes, device fingerprints, IP addresses, geography, or demographic attributes. It cannot support per-card/customer behavior, merchant risk, geographic checks, calendar-time analysis (hour of day, weekday, holidays, or seasonality), fairness analysis across demographic groups, or measurement of approval, decline, chargeback, recovery, investigation, and churn outcomes. `Amount` alone is not a complete fraud-loss measure and cannot establish the institution's false-positive or false-negative costs.

This repository does not implement a payment API, a production decision service, real-time streaming, monitoring, or operational infrastructure. The dataset cannot validate an end-to-end authorization SLA, current fraud performance, reliable concept-drift handling, or generalization to other periods, regions, issuers, or transaction channels. It covers only two days from 2013.
