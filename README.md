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



Local setup (recommended: conda)
Option A — Using conda (recommended):
# create environment
conda env create -f environment.yml
conda activate loan_rl

# install extra pip packages if needed
pip install -r requirements.txt



Option B — Using venv + pip:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt


Recommended requirements.txt
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
matplotlib>=3.5
seaborn>=0.11
torch>=2.0
d3rlpy>=0.57
shap>=0.41
gymnasium>=0.28
jupyterlab
notebook
tqdm
cloudpickle

Notes
- Install the correct torch wheel for your CUDA version (or CPU-only) using PyTorch official install instructions.
- d3rlpy may show gym / gymnasium compatibility warnings. These are often non-fatal; if you hit runtime errors, try installing gymnasium and reinstalling d3rlpy:
  pip install gymnasium
  pip install --force-reinstall d3rlpy
- If your notebook imports like from d3rlpy.metrics import DiscreteFQEConfig fail, remove or replace that import and use alternative evaluators available in your d3rlpy version (e.g., TDErrorEvaluator) — the notebook already contains a fix.

How to run the notebook end-to-end (headless)

To execute the notebook and save an executed copy:
pip install nbconvert
jupyter nbconvert --to notebook --execute \
  policy_optimization_Rajat_Satonkar_22ucs160.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --output outputs/executed_policy_optimization.ipynb

How the RL problem is framed (short technical recap)

State (s): 40 SHAP-selected features (preprocessed).

Action (a): {0: Deny, 1: Approve}.

Reward (r):

Deny → 0

Approve & Fully Paid → + loan_amnt * int_rate

Approve & Defaulted → - loan_amnt

Setting: One-step contextual bandit (each loan is an independent decision); offline RL (learn from logged dataset without live interaction).

Algorithms used / recommended: Behaviour Cloning (BC) baseline; Conservative Q-Learning (CQL) or BCQ for robust offline RL.
