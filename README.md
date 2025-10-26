# Lending_Club_Loan_Data_Project_ANN_RL

Project summary

This repository compares two approaches for loan-approval decisions:

Supervised predictive model (Task 2) — a deep-learning classifier (MLP) that predicts loan default probability. Metrics: AUC and F1-score.

Offline reinforcement learning (Task 3) — an offline RL agent that learns a policy to approve or deny loans directly from historical data (contextual bandit / one-step MDP). Metric: Estimated Policy Value (EPV) (expected financial return).

Performed EDA, preprocessing, and feature selection (top 40 SHAP features). The notebook implements training for both approaches, offline RL dataset construction, RL training (behaviour cloning baseline and conservative offline RL), offline evaluation, and a comparative analysis.


Quick start (Colab — recommended)

Open policy_optimization_Rajat_Satonkar_22ucs160.ipynb in Google Colab.

Run the setup cells to install dependencies and mount Google Drive.

Upload accepted_2007_to_2018.csv to Drive and update the notebook path (or upload to Colab).

Run cells in order (don’t skip). Sections:

Setup & installs

Data load & EDA

Preprocessing & feature selection

Supervised model training & evaluation

Offline RL (data conversion, training, evaluation)

Analysis & plots

Outputs are saved under outputs/ (models, figures); the final report PDF (if generated) appears in /content or outputs/.

If you want GPU: Runtime → Change runtime type → GPU.
