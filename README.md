# Lending_Club_Loan_Data_Project_ANN_RL

Project summary

This repository compares two approaches for loan-approval decisions:

Supervised predictive model (Task 2) — a deep-learning classifier (MLP) that predicts loan default probability. Metrics: AUC and F1-score.

Offline reinforcement learning (Task 3) — an offline RL agent that learns a policy to approve or deny loans directly from historical data (contextual bandit / one-step MDP). Metric: Estimated Policy Value (EPV) (expected financial return).

You performed EDA, preprocessing, and feature selection (top 40 SHAP features). The notebook implements training for both approaches, offline RL dataset construction, RL training (behaviour cloning baseline and conservative offline RL), offline evaluation, and a comparative analysis.
