# DQN Feature Selection — UNSW-NB15
**Project 1 for Dr. Dongjie Wang PhD Outreach**  
Single-Agent DQN for Automated Feature Selection on Network Intrusion Detection

---

## What This Does

Trains a Deep Q-Network agent to automatically discover the optimal feature subset
for intrusion detection classification on UNSW-NB15. The agent learns which of 49
features to keep or drop via trial-and-error, guided by a reward signal that balances
classification F1 against feature count.

**Target result:** DQN selects ~20-30 features with Macro F1 ≥ baseline at optimal k.

---

## One-Time Laptop Setup

### 1. Clone / place the project folder

```bash
cd ~/Desktop   # or wherever you want it
# copy the folder here, then:
cd dqn-feature-selection
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

That's it. No GPU needed — runs fine on CPU.

---

## Download the Dataset

1. Go to: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download all 4 CSV files:
   - UNSW-NB15_1.csv
   - UNSW-NB15_2.csv
   - UNSW-NB15_3.csv
   - UNSW-NB15_4.csv
3. Also download `NUSW-NB15_features.csv` (column names file)
4. Place ALL of them inside the `data/` folder

---

## Run Order

```bash
# Step 1: Merge + preprocess the 4 CSVs → saves data/processed.pkl
python src/01_preprocess.py

# Step 2: Run all baselines (MI, Chi2, RFE, full RF)
python src/02_baselines.py

# Step 3: Train DQN agent (500 episodes, ~10-20 min on CPU)
python src/03_train_dqn.py

# Step 4: Evaluate DQN best subset vs baselines → prints results table
python src/04_evaluate.py

# Step 5: Plot learning curves → saves to results/
python src/05_plots.py
```

All outputs (models, metrics, plots) go to `results/`.

---

## Results Table (fill after running)

| Method | Features Selected | Accuracy | Macro F1 |
|---|---|---|---|
| All 49 features | 49 | ? | ? |
| Mutual Information (best k) | ? | ? | ? |
| Chi-squared (best k) | ? | ? | ? |
| RFE (best k) | ? | ? | ? |
| **DQN Agent** | **learned** | **?** | **?** |

---

## GitHub Push

```bash
git init
git add .
git commit -m "DQN feature selection on UNSW-NB15 - Project 1"
git remote add origin https://github.com/sai-katari/dqn-feature-selection-unsw
git push -u origin main
```

---

## File Structure

```
dqn-feature-selection/
├── data/                   # put UNSW CSVs here
├── src/
│   ├── 01_preprocess.py    # merge, clean, normalize, save
│   ├── 02_baselines.py     # MI, Chi2, RFE, full-RF baselines
│   ├── 03_train_dqn.py     # DQN training loop
│   ├── 04_evaluate.py      # compare DQN vs baselines
│   ├── 05_plots.py         # learning curves + bar charts
│   ├── env.py              # custom Gym-style RL environment
│   ├── agent.py            # Double DQN agent + replay buffer
│   └── config.py           # all hyperparameters in one place
├── results/                # auto-created: models, metrics, plots
├── requirements.txt
└── README.md
```
