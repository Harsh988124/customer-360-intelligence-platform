# Customer 360° Intelligence & Retention Platform

An end-to-end customer intelligence platform combining data engineering, machine learning, real-time analytics, explainable AI, BI, GenAI/RAG, APIs, Docker, and MLOps.

## Business Problem

Customer data is often distributed across transactions, subscriptions, product usage, support interactions, and customer profiles.

This project creates a unified **Customer 360° view** and converts customer data into actionable retention intelligence:

- Which customers are likely to churn?
- Which customers have high lifetime value?
- How much revenue is at risk?
- Why is a customer considered risky?
- What action should the business take?
- What happens if a retention intervention succeeds?
- How can business users query customer intelligence using GenAI?

---

## Solution Overview

```text
Raw Data
   ↓
PostgreSQL
   ↓
Data Quality
   ↓
PySpark Data Engineering
   ↓
Customer 360°
   ↓
 ┌──────────────┬──────────────┬──────────────┐
 │ Segmentation │ Churn Model  │     CLV      │
 └──────────────┴──────────────┴──────────────┘
                 ↓
       Explainability / Sentiment
                 ↓
          Anomaly Detection
                 ↓
          Customer Risk Engine
                 ↓
 ┌──────────────┬──────────────┬──────────────┐
 │ Next Action  │ Revenue Risk │ Retention ROI│
 └──────────────┴──────────────┴──────────────┘
                 ↓
       Power BI / FastAPI / GenAI
                 ↓
              Docker