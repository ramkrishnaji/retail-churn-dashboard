---
title: Retail Churn Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Retail Customer Segmentation & Churn Dashboard

An end-to-end machine learning platform that segments retail customers and predicts churn risk using unsupervised and supervised learning.

## Features
1. **Unsupervised Customer Segmentation**: Categorizes customers into groups (Champions, Loyal, At-Risk, etc.) using K-Means clustering.
2. **Supervised Churn Predictor**: Predicts the likelihood of customer churn using an XGBoost model.
3. **Interactive Dashboard**: Explores segments and performs live inference using a Streamlit frontend that talks to a FastAPI backend.
