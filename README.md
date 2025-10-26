# Lending_Club_Loan_Data_Project_ANN_RL

## Project summary

This repository compares two approaches for loan-approval decisions:

Supervised predictive model (Task 2) — a deep-learning classifier (MLP) that predicts loan default probability. Metrics: AUC and F1-score.

Offline reinforcement learning (Task 3) — an offline RL agent that learns a policy to approve or deny loans directly from historical data (contextual bandit / one-step MDP). Metric: Estimated Policy Value (EPV) (expected financial return).

Performed EDA, preprocessing, and feature selection (top 40 SHAP features). The notebook implements training for both approaches, offline RL dataset construction, RL training (behaviour cloning baseline and conservative offline RL), offline evaluation, and a comparative analysis.


## Quick start (Colab — recommended)

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
### Option A — Using conda (recommended):
#create environment
conda env create -f environment.yml
conda activate loan_rl

#install extra pip packages if needed
pip install -r requirements.txt



### Option B — Using venv + pip:
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt


### Recommended requirements.txt
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

### Notes
- Install the correct torch wheel for your CUDA version (or CPU-only) using PyTorch official install instructions.
- d3rlpy may show gym / gymnasium compatibility warnings. These are often non-fatal; if you hit runtime errors, try installing gymnasium and reinstalling d3rlpy:
  pip install gymnasium
  pip install --force-reinstall d3rlpy
- If your notebook imports like from d3rlpy.metrics import DiscreteFQEConfig fail, remove or replace that import and use alternative evaluators available in your d3rlpy version (e.g., TDErrorEvaluator) — the notebook already contains a fix.

### How to run the notebook end-to-end (headless)

To execute the notebook and save an executed copy:
pip install nbconvert
jupyter nbconvert --to notebook --execute \
  policy_optimization_Rajat_Satonkar_22ucs160.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --output outputs/executed_policy_optimization.ipynb

## How the RL problem is framed (short technical recap)

State (s): 40 SHAP-selected features (preprocessed).
Action (a): {0: Deny, 1: Approve}.
Reward (r):
Deny → 0
Approve & Fully Paid → + loan_amnt * int_rate
Approve & Defaulted → - loan_amnt
Setting: One-step contextual bandit (each loan is an independent decision); offline RL (learn from logged dataset without live interaction).
Algorithms used / recommended: Behaviour Cloning (BC) baseline; Conservative Q-Learning (CQL) or BCQ for robust offline RL.

## Evaluation & metrics

- Supervised classifier: Use ROC AUC (ranking ability across thresholds) and F1-score (balance precision & recall at the chosen threshold). These measure default-prediction quality.

- RL policy: Use Estimated Policy Value (EPV) — expected average reward (net profit) when following the policy. Use off-policy evaluation (OPE) techniques for robust estimation (Importance Sampling, Doubly Robust).

- Policy comparison: Contrast classifier's implicit policy (thresholding predicted default probability) vs RL policy (explicitly maximizes EPV). Show case examples where they disagree and explain via reward arithmetic.

## Reproducibility checklist

Set random seeds (numpy, torch, random).
Save preprocessing objects (scaler, encoders) with pickle.
Save model checkpoints and hyperparameters to outputs/models/.
Freeze environment if needed: pip freeze > requirements_freeze.txt.
Example seed snippet to put at the top of the notebook:
SEED = 42
import numpy as np, random, torch
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

## Common issues & fixes

ImportError: DiscreteFQEConfig — remove or replace that import; use TDErrorEvaluator or equivalent in your d3rlpy version.
gym / gymnasium conflicts — install gymnasium and reinstall d3rlpy, or pin compatible versions.
Torch/CUDA mismatch — use the correct PyTorch wheel for your CUDA; verify torch.cuda.is_available().

## Recommended analysis & next steps (for your report)

Compute EPV with confidence intervals using robust OPE methods.
Train conservative offline RL (CQL), compare EPV vs BC and thresholded classifier policies.
Address logged-data bias (accepted-loans dataset). Consider:
Collecting outcomes for historically denied applications (if possible), or
Running a controlled pilot/A–B test (shadow mode) to gather unbiased feedback.
Add business constraints (portfolio-level limits, exposure caps).
Build explainability: SHAP for classifier; feature contributions to Q-values for RL to explain policy behavior.
Deploy cautiously: shadow mode → small pilot → monitor EPV and default rates before full rollout.

## Expected outputs (after running notebook)

outputs/models/dl_model.pth — supervised classifier weights
outputs/models/rl_cql.pkl (or similar) — offline RL agent checkpoint
outputs/figures/ — ROC, EPV comparisons, decision examples
outputs/executed_policy_optimization.ipynb — end-to-end executed notebook
final_report_policy_optimization_Rajat_Satonkar_22ucs160.pdf — concise final report (2–3 pages) summarizing Task 4 findings

## License & contact

Contact / Author: Rajat Satonkar (rajatsatonkar@gmail.com)

