# Lending Club Loan Approval: Supervised Learning vs. Offline Reinforcement Learning

Compares two approaches to automating loan-approval decisions on the [Lending Club accepted loans dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2007-2018): a supervised default-risk classifier, and an offline reinforcement-learning policy trained directly on historical outcomes.

## Approach

- **Supervised model** — an MLP classifier predicting probability of loan default, evaluated with ROC AUC and F1-score.
- **Offline RL** — the same decision reframed as a one-step contextual bandit / MDP: state = the 40 SHAP-selected features, action = {Deny, Approve}, reward = `+loan_amnt * int_rate` if approved and repaid, `-loan_amnt` if approved and defaulted, `0` if denied. A Behaviour Cloning baseline and a Conservative Q-Learning agent are trained offline (no live interaction) and evaluated by Estimated Policy Value (EPV).
- **Comparison** — the classifier's implicit threshold policy is contrasted with the RL policy's EPV-maximizing decisions, including case examples where they disagree.

Feature selection used SHAP to reduce the raw feature set to the top 40 predictors before either model was trained.

## Repository layout

- `policy_optimization.ipynb` — the full pipeline: EDA, preprocessing, feature selection, supervised training, offline RL dataset construction and training, evaluation, and comparative analysis.
- `final_report.pdf` — a short written summary of the findings.
- `requirements.txt` — Python dependencies.

## Running it

**Colab (recommended):** open `policy_optimization.ipynb` in Google Colab, run the setup cells, upload `accepted_2007_to_2018.csv` (or mount Drive), and run the notebook top to bottom. Enable a GPU runtime for faster training.

**Local:**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

To execute the notebook headlessly and save the output:
```bash
pip install nbconvert
jupyter nbconvert --to notebook --execute policy_optimization.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --output outputs/executed_policy_optimization.ipynb
```

## Evaluation metrics

| What | Metric |
|---|---|
| Supervised classifier | ROC AUC, F1-score |
| Offline RL policy | Estimated Policy Value (EPV), via off-policy evaluation (importance sampling, doubly robust) |

## Reproducibility

- Random seeds fixed (`numpy`, `random`, `torch`, `SEED = 42`).
- Preprocessing objects (scaler, encoders) and model checkpoints are saved under `outputs/`.

## Known limitations

- The dataset only contains outcomes for **accepted** loans, so the offline policy is trained under logged-data bias — it never observes what would have happened for applications that were historically denied. A fair follow-up would need either denied-application outcomes or a controlled shadow-mode pilot.
- No portfolio-level constraints (exposure caps, risk limits) are modeled; the RL reward is purely per-loan financial return.

## Troubleshooting

- `ImportError: DiscreteFQEConfig` — remove/replace with `TDErrorEvaluator` or the equivalent available in your installed `d3rlpy` version.
- `gym`/`gymnasium` conflicts — `pip install gymnasium` then `pip install --force-reinstall d3rlpy`, or pin compatible versions.
- Torch/CUDA mismatch — install the correct PyTorch wheel for your CUDA version (or CPU-only) and verify with `torch.cuda.is_available()`.

## Author

Rajat Satonkar — rajatsatonkar@gmail.com
